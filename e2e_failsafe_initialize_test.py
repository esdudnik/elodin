#!/usr/bin/env python3
"""
E2E Failsafe-from-INITIALIZE Edge Case Test.

Validates that failsafe landing works when ALTHOLD is in INITIALIZE state
(before pilot has centered/raised stick). This is the edge case where:
1. Drone climbs to ~2-3m with ALTHOLD (IN_PROGRESS)
2. ALTHOLD toggled off then on again (→ INITIALIZE)
3. Throttle stays LOW (rcThrottleAdjustment < 0, so INITIALIZE doesn't transition)
4. Failsafe triggers while ALTHOLD is still in INITIALIZE
5. BF must descend and land from INITIALIZE state

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE+ALTHOLD, 2s) → ARM(2s) → SETTLE(1s)
  → CLIMB(to ≥2m) → HOVER_BRIEF(2s)
  → DISABLE_ALTHOLD(switch off, throttle LOW)
  → REENABLE_ALTHOLD_LOW(switch on, throttle LOW, 300ms dwell — stays in INITIALIZE)
  → TRIGGER_FAILSAFE(flip AUX4 — failsafe while in INITIALIZE)
  → FAILSAFE_DESCENT(monitor BF descent) → DISARM → DONE

Run:
    cd elodin && ./run.sh e2e-failsafe-init

Prerequisites:
    Same eeprom as other tests (includes failsafe_switch_mode = STAGE2, BOXFAILSAFE on AUX4).
"""

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
SITL_EXAMPLE_DIR = SCRIPT_DIR / "examples" / "betaflight-sitl"
sys.path.insert(0, str(SITL_EXAMPLE_DIR))

import elodin as el
from comms import BetaflightSyncBridge, RCPacket, MAX_RC_CHANNELS
from config import DEFAULT_CONFIG
from sensors import IMU, SensorDataBuffer, create_sensor_system
from sim import Drone, create_physics_system

# --- Betaflight Binary ---
BETAFLIGHT_DIR = SCRIPT_DIR.parent / "betaflight"
BETAFLIGHT_PATH = BETAFLIGHT_DIR / "obj" / "main" / "betaflight_SITL.elf"

if not BETAFLIGHT_PATH.exists():
    print(f"ERROR: Betaflight SITL not found at {BETAFLIGHT_PATH}")
    sys.exit(1)


# ============================================================================
#  TEST PARAMETERS
# ============================================================================

CLIMB_MIN_ALT = 2.0             # meters — minimum altitude before triggering sequence
HOVER_BRIEF_DURATION = 2.0      # seconds — stabilize before toggling
DISABLE_ALTHOLD_DURATION = 0.2  # seconds — ALTHOLD off dwell (short to preserve altitude)
REENABLE_DWELL = 0.3            # seconds — ALTHOLD on below deadband (stay in INITIALIZE)
REENABLE_VZ_DESCENT_LIMIT = -0.3  # m/s — hard fail if drone descending before failsafe trigger
FAILSAFE_TRIGGER_MIN_ALT = 1.5  # meters — must still be above this when failsafe triggers

# Throttle during toggle phases: below 1250 "low throttle" threshold.
# Assumes altHoldThrottleType = STICK (default). At 1200, BF detects
# throttleIsLow=true (< 25% = 1250), so setupAltitudeController() falls
# back to mid-stick zero-point (1500). With deadband=50:
#   rcThrottleAdjustment = applyDeadband(1200-1500, 50) = -250
# Clearly negative → ALTHOLD stays in INITIALIZE.
TOGGLE_THROTTLE = 1200

FAILSAFE_DESCENT_TIMEOUT = 30.0
BF_DISARM_MOTOR_THRESHOLD = 0.01
BF_DISARM_DWELL = 0.3

TEST_TIMEOUT = 90.0

# RC channels
CH_ROLL = 0
CH_PITCH = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_ARM = 4
CH_ANGLE = 5
CH_ALTHOLD = 6
CH_FAILSAFE = 7

RC_CENTER = 1500
RC_LOW = 1000
MODE_ON = 1800
MODE_OFF = 1000
ALTHOLD_CLIMB = 1700
ALTHOLD_HOLD = 1500

BOOT_DURATION = 5.0
ARM_DURATION = 2.0
ALTHOLD_SETTLE = 2.0
DISARM_DURATION = 1.0
CLIMB_TIMEOUT = 30.0
ALTITUDE_CEILING = 50.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    SETTLE = auto()
    CLIMB = auto()
    HOVER_BRIEF = auto()
    DISABLE_ALTHOLD = auto()       # toggle ALTHOLD off
    REENABLE_ALTHOLD_LOW = auto()  # toggle ALTHOLD on with throttle LOW (INITIALIZE state)
    TRIGGER_FAILSAFE = auto()      # flip BOXFAILSAFE — failsafe while in INITIALIZE
    FAILSAFE_DESCENT = auto()      # monitor BF descent
    DISARM = auto()
    DONE = auto()


@dataclass
class TestState:
    phase: Phase = Phase.BOOT
    phase_start_time: float = 0.0
    sim_time: float = 0.0
    tick: int = 0

    motors: np.ndarray = field(default_factory=lambda: np.zeros(4))
    max_motor: float = 0.0
    step_count: int = 0

    current_altitude: float = 0.0
    current_vz: float = 0.0
    max_altitude: float = 0.0
    baro_altitude: float = 0.0

    hover_target: float = 0.0

    # INITIALIZE proxy check
    reenable_max_vz: float = 0.0  # max |vz| during REENABLE dwell — should stay near zero

    # Failsafe tracking
    failsafe_trigger_alt: float = 0.0
    failsafe_descent_started: bool = False
    failsafe_descent_time: float = 0.0

    land_altitude: float = 0.0
    land_velocity: float = 0.0
    bf_disarm_dwell: float = 0.0

    crash_detected: bool = False
    crash_reason: str = ""

    last_print_time: float = -1.0
    results_printed: bool = False
    test_passed: bool = False
    quat_xyzw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    gyro_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>14}] --> [{new_phase.name}]")

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
    channels[CH_FAILSAFE] = MODE_OFF

    phase = state.phase

    if phase == Phase.BOOT:
        pass

    elif phase == Phase.PRESELECT:
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.ARM:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.SETTLE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.CLIMB:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_CLIMB

    elif phase == Phase.HOVER_BRIEF:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.DISABLE_ALTHOLD:
        # ALTHOLD off, throttle below deadband but not idle (preserves altitude in ANGLE mode)
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_OFF
        channels[CH_THROTTLE] = TOGGLE_THROTTLE

    elif phase == Phase.REENABLE_ALTHOLD_LOW:
        # ALTHOLD back on, throttle below deadband → ALTHOLD enters INITIALIZE
        # rcThrottleAdjustment < 0 → stays in INITIALIZE (doesn't transition to IN_PROGRESS)
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = TOGGLE_THROTTLE

    elif phase == Phase.TRIGGER_FAILSAFE:
        # Same as REENABLE but with BOXFAILSAFE on
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = TOGGLE_THROTTLE
        channels[CH_FAILSAFE] = MODE_ON

    elif phase == Phase.FAILSAFE_DESCENT:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = TOGGLE_THROTTLE
        channels[CH_FAILSAFE] = MODE_ON

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW
        channels[CH_FAILSAFE] = MODE_OFF

    return channels


# ============================================================================
#  PHASE TRANSITIONS
# ============================================================================

def update_phase(state: TestState, t: float, dt: float):
    phase = state.phase
    elapsed = state.phase_elapsed(t)

    if state.current_altitude > ALTITUDE_CEILING:
        state.crash_detected = True
        state.crash_reason = f"Altitude ceiling exceeded ({state.current_altitude:.1f}m)"
        state.transition(Phase.DISARM, t)
        return

    if phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.transition(Phase.PRESELECT, t)

    elif phase == Phase.PRESELECT:
        if elapsed >= ALTHOLD_SETTLE:
            print(f"[{t:6.1f}s] [     PRESELECT] Switches set. Arming...")
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(f"[{t:6.1f}s] [           ARM] Armed. alt={state.current_altitude:.1f}m")
            state.transition(Phase.SETTLE, t)

    elif phase == Phase.SETTLE:
        if elapsed >= 1.0:
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        if state.current_altitude >= CLIMB_MIN_ALT:
            state.hover_target = state.current_altitude
            print(f"[{t:6.1f}s] [         CLIMB] Reached {state.current_altitude:.1f}m. Brief hover...")
            state.transition(Phase.HOVER_BRIEF, t)
        elif elapsed >= CLIMB_TIMEOUT:
            print(f"[{t:6.1f}s] [         CLIMB] TIMEOUT at {state.current_altitude:.1f}m")
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOVER_BRIEF:
        if elapsed >= HOVER_BRIEF_DURATION:
            print(
                f"[{t:6.1f}s] [   HOVER_BRIEF] "
                f"Stable at {state.current_altitude:.1f}m. Disabling ALTHOLD..."
            )
            state.transition(Phase.DISABLE_ALTHOLD, t)

    elif phase == Phase.DISABLE_ALTHOLD:
        if elapsed >= DISABLE_ALTHOLD_DURATION:
            print(
                f"[{t:6.1f}s] [DISABLE_ALTHOLD] "
                f"ALTHOLD off for {DISABLE_ALTHOLD_DURATION}s. "
                f"Re-enabling with LOW throttle (INITIALIZE state)..."
            )
            state.reenable_max_vz = 0.0
            state.transition(Phase.REENABLE_ALTHOLD_LOW, t)

    elif phase == Phase.REENABLE_ALTHOLD_LOW:
        # Proxy check: hard fail if drone is descending during INITIALIZE dwell.
        # Upward transients from throttle change are OK — they don't indicate wrong FSM state.
        # Descent before failsafe trigger means ALTHOLD INITIALIZE isn't holding.
        if state.current_vz < REENABLE_VZ_DESCENT_LIMIT:
            print(
                f"[{t:6.1f}s] [REENABLE_LOW] "
                f"FAIL: drone descending (vz={state.current_vz:.2f}m/s) during INITIALIZE dwell. "
                f"ALTHOLD may not be in INITIALIZE."
            )
            state.crash_detected = True
            state.crash_reason = f"Descent during INITIALIZE dwell (vz={state.current_vz:.2f}m/s)"
            state.transition(Phase.DISARM, t)
            return

        if elapsed >= REENABLE_DWELL:
            state.failsafe_trigger_alt = state.current_altitude
            # Hard fail if altitude dropped too low
            if state.current_altitude < FAILSAFE_TRIGGER_MIN_ALT:
                print(
                    f"[{t:6.1f}s] [REENABLE_LOW] "
                    f"FAIL: altitude too low ({state.current_altitude:.1f}m < {FAILSAFE_TRIGGER_MIN_ALT}m) "
                    f"for meaningful failsafe descent test."
                )
                state.crash_detected = True
                state.crash_reason = f"Altitude too low at failsafe trigger ({state.current_altitude:.1f}m)"
                state.transition(Phase.DISARM, t)
                return
            print(
                f"[{t:6.1f}s] [REENABLE_LOW] "
                f"ALTHOLD in INITIALIZE for {REENABLE_DWELL}s. "
                f"alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s. "
                f"Triggering failsafe..."
            )
            state.transition(Phase.TRIGGER_FAILSAFE, t)

    elif phase == Phase.TRIGGER_FAILSAFE:
        print(
            f"[{t:6.1f}s] [TRIGGER_FAILSAFE] "
            f"BOXFAILSAFE ON at alt={state.current_altitude:.2f}m "
            f"(ALTHOLD in INITIALIZE). Monitoring descent..."
        )
        state.transition(Phase.FAILSAFE_DESCENT, t)

    elif phase == Phase.FAILSAFE_DESCENT:
        if not state.failsafe_descent_started and state.current_vz < -0.3:
            state.failsafe_descent_started = True
            print(
                f"[{t:6.1f}s] [FAILSAFE_DESCENT] "
                f"Descent confirmed: vz={state.current_vz:.2f}m/s alt={state.current_altitude:.2f}m"
            )

        all_motors_zero = all(m < BF_DISARM_MOTOR_THRESHOLD for m in state.motors)
        if all_motors_zero:
            state.bf_disarm_dwell += dt
        else:
            state.bf_disarm_dwell = 0.0

        if state.bf_disarm_dwell >= BF_DISARM_DWELL:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            state.failsafe_descent_time = elapsed
            print(
                f"[{t:6.1f}s] [FAILSAFE_DESCENT] "
                f"BF failsafe disarm: alt={state.current_altitude:.2f}m "
                f"descent_time={elapsed:.1f}s (from {state.failsafe_trigger_alt:.1f}m)"
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= FAILSAFE_DESCENT_TIMEOUT:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            state.failsafe_descent_time = elapsed
            print(
                f"[{t:6.1f}s] [FAILSAFE_DESCENT] "
                f"FAIL: disarm not observed after {FAILSAFE_DESCENT_TIMEOUT:.0f}s"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.DISARM:
        if elapsed >= DISARM_DURATION:
            state.transition(Phase.DONE, t)


# ============================================================================
#  STATUS + RESULTS
# ============================================================================

def print_status(state: TestState, t: float):
    if int(t) <= state.last_print_time:
        return
    state.last_print_time = int(t)

    motors_str = ",".join(f"{m:.3f}" for m in state.motors)
    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]
    althold_on = channels[CH_ALTHOLD] == MODE_ON
    failsafe_on = channels[CH_FAILSAFE] == MODE_ON

    if throttle <= 1050:
        stick = "IDLE"
    elif throttle >= 1650:
        stick = "CLIMB"
    else:
        stick = "HOLD"

    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    alt = "ALT" if althold_on else "---"
    fs = " FS" if failsafe_on else ""

    print(
        f"[{t:6.1f}s] [{state.phase.name:>14}] "
        f"alt={state.current_altitude:+7.2f}m vz={state.current_vz:+5.2f}m/s "
        f"motors=[{motors_str}] T={throttle} ({stick}) modes=[{arm}|{alt}]{fs}"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E FAILSAFE-FROM-INITIALIZE TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print()
    print(f"  --- Failsafe from INITIALIZE ---")
    print(f"  Trigger altitude:     {state.failsafe_trigger_alt:.1f}m")
    print(f"  INIT dwell no descent: {'Yes' if not state.crash_detected else 'No'}")
    print(f"  Descent started:      {'Yes' if state.failsafe_descent_started else 'No'}")
    print(f"  Descent time:         {state.failsafe_descent_time:.1f}s (timeout: {FAILSAFE_DESCENT_TIMEOUT:.0f}s)")
    print(f"  Landing altitude:     {state.land_altitude:.2f}m")
    print(f"  Landing velocity:     {state.land_velocity:.2f}m/s")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if not state.failsafe_descent_started:
        passed = False
        issues.append("Failsafe descent never started from INITIALIZE state")

    if state.failsafe_descent_time >= FAILSAFE_DESCENT_TIMEOUT:
        passed = False
        issues.append(f"Failsafe disarm not observed within {FAILSAFE_DESCENT_TIMEOUT:.0f}s")

    if state.failsafe_trigger_alt < FAILSAFE_TRIGGER_MIN_ALT and state.failsafe_trigger_alt > 0:
        passed = False
        issues.append(
            f"Trigger altitude too low ({state.failsafe_trigger_alt:.1f}m < {FAILSAFE_TRIGGER_MIN_ALT}m)"
        )

    if state.land_altitude > 0.20:
        passed = False
        issues.append(f"Did not land properly ({state.land_altitude:.2f}m)")

    if state.step_count == 0:
        passed = False

        issues.append("No motor responses")

    state.test_passed = passed


    if passed and not issues:
        print("  Status:               PASS")
    elif passed:
        print("  Status:               PASS (with warnings)")
        for issue in issues:
            print(f"    WARNING: {issue}")
    else:
        print("  Status:               FAIL")
        for issue in issues:
            print(f"    FAIL: {issue}")

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
        hsplit name = "Failsafe INIT Test" {
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
    "e2e-failsafe-init-test.kdl",
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
            print("  E2E FAILSAFE-FROM-INITIALIZE Test")
            print(f"  Scenario: climb to ≥{CLIMB_MIN_ALT}m, toggle ALTHOLD, "
                  f"trigger failsafe during INITIALIZE")
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
        s.baro_altitude = float(baro[0]) if len(baro) > 0 else s.current_altitude
        s.quat_xyzw = world_pos[:4]
        s.gyro_body = gyro
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
        s.max_motor = max(s.max_motor, float(np.max(s.motors)))
        s.step_count += 1
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        if s.phase not in (Phase.BOOT, Phase.DONE):
            print(f"[{t:6.1f}s] WARNING: Motor timeout")

    print_status(s, t)

    if s.phase == Phase.DONE and not s.results_printed:
        b.stop()
        elapsed = time.time() - _start_time[0]
        print(f"\nSimulation: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)")
        print_results(s)

    if tick >= max_ticks - 1 and not s.results_printed:
        print(f"\n[{t:6.1f}s] TEST TIMEOUT")
        s.land_altitude = s.current_altitude
        s.land_velocity = abs(s.current_vz)
        b.stop()
        print_results(s)


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv
db_path = "/tmp/e2e_failsafe_init_test_db" if running_under_editor else "e2e_failsafe_init_test_db"

print(f"E2E FAILSAFE-FROM-INITIALIZE Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: climb → toggle ALTHOLD → failsafe during INITIALIZE")

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

# Exit with test result code
sys.exit(0 if _state[0] and _state[0].test_passed else 1)
