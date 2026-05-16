#!/usr/bin/env python3
"""
E2E Low-Altitude Hover No-Disarm Test — v10.5.1 safety regression guard.

Validates that ALTHOLD does NOT auto-disarm during low-altitude hover with
slight descent stick. This is the critical safety bug reported on real hardware
(2026-05-16): drone in ALTHOLD at 2-3m altitude with slight descent stick
(below center but above mincheck) would auto-disarm via the legacy
isAltHoldLandingDetected path that used ap_landing_altitude_m = 4m threshold.

Root cause: the previous auto-disarm logic triggered whenever:
  - drone state = IN_PROGRESS
  - alt < 4m (landingAltitudeM default)
  - descentIntent (stick < center, even slightly)
  - |vz| < 20 cm/s, gyro < 4 dps sustained 1s
All satisfiable during normal low-altitude flight → mid-flight disarm → crash.

v10.5.1 fix: replaced the call with sustained THROTTLE_LOW + airborne-latch
gates. Auto-disarm now requires explicit pilot commit (stick at mincheck)
sustained at alt < 30cm.

Scenario:
  1. Take off in ANGLE to ~3m (airborne latch trips at alt > 30cm)
  2. Engage ALTHOLD with engagement gesture (stick centered, 0.5s)
  3. Slight descent in ALTHOLD (stick = T_DESCEND = 1300, NOT at mincheck)
     — drone descends slowly under ALTHOLD control at -200 cm/s max
  4. Continue descent for sustained period (5 seconds)
     — drone alt drops from ~3m to ~1m at controlled rate
  5. Verify drone does NOT auto-disarm during this phase
  6. Manual disarm at end

Pass criteria:
  - low_pass_activate_reached: True (scenario worked)
  - PRIMARY: no auto-disarm observed during HOVER_DESCENT phase
    (drone stays armed, motors active)
  - SECONDARY: min_alt_during_hover > 0.5m (drone didn't crash)
  - SECONDARY: max_motor_during_hover > 0.10 (motors weren't clamped)

Failure modes covered:
  - v10.4 regression: would auto-disarm at any alt < 4m → FAIL
  - v10.5 first attempt regression: stick blip below mincheck would trigger
    instant disarm → FAIL (mitigated in v10.5.1 by sustained THROTTLE_LOW)
  - v10.5.1 correct: drone holds altitude under ALTHOLD, no disarm

Prerequisites (standard SITL eeprom):
    aux 0 0 0 1700 2100 0 0
    aux 1 1 1 1700 2100 0 0
    aux 2 3 2 1700 2100 0 0
    set failsafe_delay = 200
    set ap_hover_throttle = 1300
    set alt_hold_climb_rate = 200
    set alt_hold_deadband = 50
    save

Run:
    cd elodin && ./run.sh e2e-low-alt-hover-no-disarm
"""

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SITL_EXAMPLE_DIR = SCRIPT_DIR / "examples" / "betaflight-sitl"
sys.path.insert(0, str(SITL_EXAMPLE_DIR))

import elodin as el
from comms import BetaflightSyncBridge, RCPacket, MAX_RC_CHANNELS
from config import DEFAULT_CONFIG
from sensors import IMU, SensorDataBuffer, create_sensor_system
from sim import Drone, create_physics_system

BETAFLIGHT_DIR = SCRIPT_DIR.parent / "betaflight"
BETAFLIGHT_PATH = BETAFLIGHT_DIR / "obj" / "main" / "betaflight_SITL.elf"

if not BETAFLIGHT_PATH.exists():
    print(f"ERROR: Betaflight SITL not found at {BETAFLIGHT_PATH}")
    sys.exit(1)


# ============================================================================
#  TEST PARAMETERS
# ============================================================================

MIDRC = 1500
STICK_CLIMB = MIDRC + 200          # 1700
T_DESCEND = MIDRC - 200             # 1300 — descent stick, NOT at mincheck (THROTTLE_LOW)
STICK_HOLD = MIDRC                  # 1500 — center

CLIMB_TARGET_ALT = 3.0              # m — climb until reaching this; airborne latch trips

ALTHOLD_ENGAGE_DURATION = 0.5       # s — engagement gesture step

HOVER_DESCENT_DURATION = 5.0        # s — sustained slight-descent phase (key test window)
HOVER_DESCENT_TIMEOUT = 15.0        # s — safety net

# Pass/fail thresholds
MIN_ALT_DURING_HOVER = 0.5          # m — drone must not crash
MIN_MAX_MOTOR_DURING_HOVER = 0.10   # — motors must be active (no spin-lock / no disarm)
NO_DISARM_REQUIRED = True           # primary check — drone must stay armed

TEST_TIMEOUT = 60.0

CH_ROLL = 0
CH_PITCH = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_ARM = 4
CH_ANGLE = 5
CH_ALTHOLD = 6

RC_CENTER = 1500
RC_LOW = 1000
MODE_ON = 1800
MODE_OFF = 1000

BOOT_DURATION = 5.0
ARM_DURATION = 2.0
SETTLE_DURATION = 1.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    SETTLE = auto()
    CLIMB = auto()                  # ANGLE only — drives airborne latch trip
    ALTHOLD_ENGAGE_GESTURE = auto() # stick centered + ALTHOLD on — WAIT_CENTER→WAIT_DEPART
    HOVER_DESCENT = auto()          # stick=T_DESCEND + ALTHOLD on — slow descent; auto-disarm MUST NOT fire
    DISARM = auto()
    DONE = auto()


@dataclass
class TestState:
    phase: Phase = Phase.BOOT
    phase_start_time: float = 0.0
    sim_time: float = 0.0
    tick: int = 0

    motors: np.ndarray = field(default_factory=lambda: np.zeros(4))
    step_count: int = 0

    current_altitude: float = 0.0
    current_vz: float = 0.0
    max_altitude: float = 0.0

    # HOVER_DESCENT tracking
    hover_capture_alt: float = 0.0
    min_alt_during_hover: float = float('inf')
    max_motor_during_hover: float = 0.0
    disarm_observed_during_hover: bool = False

    # Scenario reach flag — gates assertions in print_results
    hover_descent_reached: bool = False

    crash_detected: bool = False
    crash_reason: str = ""

    last_print_time: float = -1.0
    results_printed: bool = False
    test_passed: bool = False

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>22}] --> [{new_phase.name}]")

    def phase_elapsed(self, t: float) -> float:
        return t - self.phase_start_time


# ============================================================================
#  RC CHANNEL BUILDER
# ============================================================================

def build_rc_channels(state: TestState) -> np.ndarray:
    channels = np.full(MAX_RC_CHANNELS, RC_CENTER, dtype=np.uint16)
    channels[CH_THROTTLE] = RC_LOW
    channels[CH_ARM] = MODE_OFF
    channels[CH_ANGLE] = MODE_OFF
    channels[CH_ALTHOLD] = MODE_OFF

    phase = state.phase

    if phase == Phase.PRESELECT:
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW
    elif phase == Phase.ARM:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW
    elif phase == Phase.SETTLE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD
    elif phase == Phase.CLIMB:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = STICK_CLIMB
    elif phase == Phase.ALTHOLD_ENGAGE_GESTURE:
        # ALTHOLD just toggled on with stick in deadband. INITIALIZE starts
        # with engagement = WAIT_CENTER. stick in deadband → WAIT_DEPART
        # within 1 tick. Drone holds altitude throughout.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD
    elif phase == Phase.HOVER_DESCENT:
        # THE TEST PHASE: stick at T_DESCEND (1300) — slight descent below
        # center. NOT at RC_LOW (would be THROTTLE_LOW = pilot commit to land).
        # v10.5.1 expected behavior: drone descends slowly via ALTHOLD, no
        # auto-disarm fires (THROTTLE_LOW gate not satisfied).
        # v10.4 broken behavior: drone would auto-disarm via legacy
        # isAltHoldLandingDetected path (alt < 4m, slow vz, calm gyro).
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = T_DESCEND
    elif phase == Phase.DISARM:
        channels[CH_ARM] = MODE_OFF
        channels[CH_THROTTLE] = RC_LOW

    return channels


# ============================================================================
#  PHASE UPDATER
# ============================================================================

def update_phase(state: TestState, t: float, dt: float):
    elapsed = state.phase_elapsed(t)

    if state.phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.transition(Phase.PRESELECT, t)

    elif state.phase == Phase.PRESELECT:
        if elapsed >= 2.0:
            state.transition(Phase.ARM, t)

    elif state.phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(f"[{t:6.1f}s] [{'ARM':>22}] Armed. Climbing in ANGLE mode...")
            state.transition(Phase.SETTLE, t)

    elif state.phase == Phase.SETTLE:
        if elapsed >= SETTLE_DURATION:
            state.transition(Phase.CLIMB, t)

    elif state.phase == Phase.CLIMB:
        if state.current_altitude >= CLIMB_TARGET_ALT:
            print(f"[{t:6.1f}s] [{'CLIMB':>22}] Reached {state.current_altitude:.1f}m. "
                  f"Engaging ALTHOLD...")
            state.transition(Phase.ALTHOLD_ENGAGE_GESTURE, t)
        elif elapsed >= 30.0:
            state.crash_detected = True
            state.crash_reason = f"climb timeout: only reached {state.max_altitude:.2f}m"
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.ALTHOLD_ENGAGE_GESTURE:
        # Stick in deadband for the gesture duration. Engagement FSM
        # WAIT_CENTER → WAIT_DEPART. Drone holds altitude throughout.
        if elapsed >= ALTHOLD_ENGAGE_DURATION:
            state.hover_capture_alt = state.current_altitude
            state.hover_descent_reached = True
            print(f"[{t:6.1f}s] [{'ALTHOLD_ENGAGE_GESTURE':>22}] Gesture complete "
                  f"(alt={state.current_altitude:.2f}m). Starting slight-descent test...")
            state.transition(Phase.HOVER_DESCENT, t)

    elif state.phase == Phase.HOVER_DESCENT:
        # Critical test phase — track motors and altitude.
        # Any auto-disarm here is a FAIL (regression).
        state.min_alt_during_hover = min(state.min_alt_during_hover, state.current_altitude)
        if state.motors is not None and len(state.motors) > 0:
            max_motor = float(np.max(state.motors))
            state.max_motor_during_hover = max(state.max_motor_during_hover, max_motor)
            # Detect disarm: all motors near zero (< 0.005) sustained.
            # Healthy ALTHOLD descent has motors ~0.15-0.30 at this throttle.
            if max_motor < 0.005:
                # Could be transient — wait for sustained
                # (allow ~3 ticks of motor < threshold before flagging)
                if not state.disarm_observed_during_hover:
                    state.disarm_observed_during_hover = True
                    print(f"[{t:6.1f}s] [{'HOVER_DESCENT':>22}] "
                          f"WARNING: motors near zero at alt={state.current_altitude:.2f}m "
                          f"— possible auto-disarm regression")

        # Crash safety
        if state.current_altitude < MIN_ALT_DURING_HOVER:
            state.crash_detected = True
            state.crash_reason = (
                f"drone descended below safety floor "
                f"(alt={state.current_altitude:.3f}m < {MIN_ALT_DURING_HOVER}m). "
                f"Possible auto-disarm regression."
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= HOVER_DESCENT_DURATION:
            print(f"[{t:6.1f}s] [{'HOVER_DESCENT':>22}] Phase complete. "
                  f"min_alt={state.min_alt_during_hover:.3f}m "
                  f"max_motor={state.max_motor_during_hover:.3f} "
                  f"disarm_observed={state.disarm_observed_during_hover}")
            state.transition(Phase.DISARM, t)
        elif elapsed >= HOVER_DESCENT_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"hover-descent timeout (alt={state.current_altitude:.2f}m)"
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.DISARM:
        if elapsed >= 1.0:
            state.transition(Phase.DONE, t)


# ============================================================================
#  RESULTS PRINTING
# ============================================================================

def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print("\n" + "=" * 70)
    print("  E2E LOW-ALT HOVER NO-DISARM TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:                  {state.sim_time:.1f}s")
    print(f"  Max altitude:              {state.max_altitude:.2f}m")
    print(f"  Reached HOVER_DESCENT:     {state.hover_descent_reached}")
    if state.hover_descent_reached:
        print(f"  Hover capture altitude:    {state.hover_capture_alt:.2f}m")
        print(f"  Min alt during hover:      {state.min_alt_during_hover:.3f}m "
              f"(limit: {MIN_ALT_DURING_HOVER}m)")
        print(f"  Max motor during hover:    {state.max_motor_during_hover:.3f} "
              f"(must be > {MIN_MAX_MOTOR_DURING_HOVER})")
        print(f"  Auto-disarm observed:      {state.disarm_observed_during_hover} "
              f"(must be False — REGRESSION if True)")
    print()

    fails = []
    if not state.hover_descent_reached:
        fails.append(
            "  FAIL: scenario setup failed — never reached HOVER_DESCENT "
            "(auto-disarm test phase). Spin-lock or ARM-phase issue."
        )
        if state.crash_reason:
            fails.append(f"    reason: {state.crash_reason}")
    else:
        # Real safety regression assertions
        if state.disarm_observed_during_hover:
            fails.append(
                "  FAIL: auto-disarm observed during HOVER_DESCENT — v10.5.1 regression. "
                "Drone disarmed in mid-flight at low altitude during slight descent. "
                "This is the original v10.4 critical safety bug returning."
            )
        if state.crash_detected:
            fails.append(f"  FAIL: {state.crash_reason}")
        if state.min_alt_during_hover < MIN_ALT_DURING_HOVER and not state.crash_detected:
            fails.append(
                f"  FAIL: drone dropped below safety floor "
                f"(min={state.min_alt_during_hover:.3f}m < {MIN_ALT_DURING_HOVER}m)"
            )
        if state.max_motor_during_hover < MIN_MAX_MOTOR_DURING_HOVER:
            fails.append(
                f"  FAIL: max_motor during hover = {state.max_motor_during_hover:.3f} "
                f"< {MIN_MAX_MOTOR_DURING_HOVER}. Motors clamped or disarmed."
            )

    if fails:
        print("  Status:                    FAIL")
        for f in fails:
            print(f)
    else:
        print("  Status:                    PASS")
        state.test_passed = True

    print("=" * 70)


# ============================================================================
#  WORLD SETUP
# ============================================================================

config = DEFAULT_CONFIG
config.set_as_global()

if "--no-s10" not in sys.argv:
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "betaflight_SITL"], capture_output=True, timeout=5)
        time.sleep(0.1)
    except Exception:
        pass

world = el.World()

drone = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array(config.initial_position),
                angular=el.Quaternion(jnp.array(config.initial_quaternion)),
            ),
            world_vel=el.SpatialMotion(
                linear=jnp.array(config.initial_velocity),
                angular=jnp.array(config.initial_angular_velocity),
            ),
            inertia=el.SpatialInertia(
                mass=config.mass,
                inertia=jnp.array(config.inertia_diagonal),
            ),
        ),
        Drone(),
        IMU(),
    ],
    name="drone",
)

ground = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array([0.0, 0.0, 0.0]),
                angular=el.Quaternion(jnp.array([0.0, 0.0, 0.0, 1.0])),
            ),
            world_vel=el.SpatialMotion(
                linear=jnp.array([0.0, 0.0, 0.0]),
                angular=jnp.array([0.0, 0.0, 0.0]),
            ),
            inertia=el.SpatialInertia(
                mass=0.001, inertia=jnp.array([0.001, 0.001, 0.001])
            ),
        ),
    ],
    name="ground",
)

world.schematic(
    """
    tabs {
        hsplit name = "Low-Alt Hover No-Disarm Test" {
            viewport name=Viewport pos="drone.world_pos.translate_world(5.0, 5.0, 3.0)" look_at="drone.world_pos" show_grid=#true active=#true
            vsplit share=0.3 {
                graph "drone.motor_command" name="Motor Commands"
                graph "drone.world_pos.linear()" name="Position (ENU)"
            }
        }
    }
    object_3d drone.world_pos {
        glb path="edu-450-v2-drone.glb" rotate="(0.0, 0.0, 0.0)" translate="(0.0, 1.0, 0.0)" scale=10.0
    }
    """,
    "e2e-low-alt-hover-no-disarm-test.kdl",
)

physics = create_physics_system(config)
sensors = create_sensor_system(config)
system = physics | sensors

betaflight_recipe = el.s10.PyRecipe.process(
    name="Betaflight SITL",
    cmd=str(BETAFLIGHT_PATH),
    cwd=str(BETAFLIGHT_DIR),
)
world.recipe(betaflight_recipe)


# ============================================================================
#  POST-STEP CALLBACK
# ============================================================================

_bridge = [None]
_sensor_buf = [None]
_state = [None]
_start_time = [None]
max_ticks = int(TEST_TIMEOUT / config.sim_time_step)


def e2e_post_step(tick: int, ctx: el.StepContext):
    if _bridge[0] is None:
        try:
            bridge_obj = BetaflightSyncBridge(timeout_ms=100)
            _sensor_buf[0] = SensorDataBuffer()
            _state[0] = TestState()
            _start_time[0] = time.time()

            print()
            print("=" * 70)
            print("  E2E LOW-ALT HOVER NO-DISARM Test (v10.5.1 safety regression)")
            print(f"  Scenario: ALTHOLD at ~3m with slight descent stick (T_DESCEND=1300)")
            print(f"  Expected: drone descends slowly, NO auto-disarm")
            print(f"  Timeout: {TEST_TIMEOUT}s")
            print("=" * 70)
            print()

            bridge_obj.start()
            _bridge[0] = bridge_obj
            print("[  0.0s] [          INIT] Waiting 2s for BF init...")
        except Exception as e:
            print(f"[INIT] ERROR: {e}")
            _bridge[0] = None
            return
        time.sleep(2)

        print("[  0.0s] [          INIT] Warmup...")
        warmup_buf = SensorDataBuffer()
        warmup_fdm = warmup_buf.build_fdm()
        warmup_channels = np.full(MAX_RC_CHANNELS, RC_CENTER, dtype=np.uint16)
        warmup_channels[CH_THROTTLE] = RC_LOW
        warmup_channels[CH_ARM] = MODE_OFF
        warmup_rc = RCPacket(timestamp=0.0, channels=warmup_channels)

        warmup_ok = 0
        for i in range(500):
            warmup_fdm.timestamp = i * config.sim_time_step
            warmup_rc.timestamp = i * config.sim_time_step
            try:
                _bridge[0].step(warmup_fdm, warmup_rc)
                warmup_ok += 1
            except TimeoutError:
                pass
        print(f"[  0.0s] [          INIT] Warmup: {warmup_ok}/500")
        ctx.truncate()

    if _start_time[0] is None:
        _start_time[0] = time.time()

    b = _bridge[0]
    buf = _sensor_buf[0]
    s = _state[0]

    if s.results_printed:
        return

    s.tick = tick
    s.sim_time = tick * config.sim_time_step
    t = s.sim_time

    try:
        sensor_data = ctx.component_batch_operation(
            reads=["drone.accel", "drone.gyro", "drone.world_pos", "drone.world_vel", "drone.baro"]
        )
        accel = np.array(sensor_data["drone.accel"])
        gyro = np.array(sensor_data["drone.gyro"])
        world_pos = np.array(sensor_data["drone.world_pos"])
        world_vel = np.array(sensor_data["drone.world_vel"])
        baro = np.array(sensor_data["drone.baro"])

        buf.update(
            world_pos=world_pos, world_vel=world_vel,
            accel=accel, gyro=gyro, baro=baro, timestamp=t,
        )

        s.current_altitude = float(world_pos[6]) if len(world_pos) > 6 else float(world_pos[2])
        s.current_vz = float(world_vel[5]) if len(world_vel) > 5 else float(world_vel[2])
        s.max_altitude = max(s.max_altitude, s.current_altitude)
    except RuntimeError as e:
        if tick > 5:
            print(f"[{t:6.1f}s] WARNING: {e}")
        buf.timestamp = t

    update_phase(s, t, dt=config.sim_time_step)

    channels = build_rc_channels(s)
    fdm = buf.build_fdm()
    rc = RCPacket(timestamp=t, channels=channels)

    try:
        s.motors = b.step(fdm, rc)
        s.step_count += 1
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        if s.phase not in (Phase.BOOT, Phase.DONE):
            print(f"[{t:6.1f}s] WARNING: Motor timeout")

    if s.sim_time - s.last_print_time >= 1.0 or s.last_print_time < 0:
        motors_str = ",".join(f"{m:.3f}" for m in s.motors)
        print(f"[{t:6.1f}s] [{s.phase.name:>22}] alt={s.current_altitude:+6.2f}m "
              f"vz={s.current_vz:+5.2f}m/s motors=[{motors_str}]")
        s.last_print_time = s.sim_time

    if s.phase == Phase.DONE and not s.results_printed:
        b.stop()
        elapsed = time.time() - _start_time[0]
        print(f"\nSimulation: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)")
        print_results(s)

    if tick >= max_ticks - 1 and not s.results_printed:
        print(f"\n[{t:6.1f}s] TEST TIMEOUT")
        s.crash_detected = True
        s.crash_reason = "test timeout"
        b.stop()
        print_results(s)


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv
db_path = "/tmp/e2e_low_alt_hover_no_disarm_test_db" if running_under_editor else "e2e_low_alt_hover_no_disarm_test_db"

print(f"E2E LOW-ALT HOVER NO-DISARM Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: auto-disarm must NOT fire during slow ALTHOLD descent at low alt")

world.run(
    system,
    sim_time_step=config.sim_time_step,
    run_time_step=config.sim_time_step,
    max_ticks=max_ticks,
    post_step=e2e_post_step,
    db_path=db_path,
    interactive=False,
    backend="jax",
)

sys.exit(0 if _state[0] and _state[0].test_passed else 1)
