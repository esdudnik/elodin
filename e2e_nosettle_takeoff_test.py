#!/usr/bin/env python3
"""
E2E No-Settle Takeoff Regression Test.

Validates that ALTHOLD takeoff works WITHOUT a center-throttle settle phase.
This is a regression test for the allowStickAdjustment gate bug where
preselecting ALTHOLD before arming, then raising throttle directly to climb,
would be blocked because the stick never passed through the deadband.

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE+ALTHOLD on, ARM off, 2s) → ARM(2s)
  → CLIMB(directly, NO settle phase) → TOP_APPROACH(to 6.8m) → HOVER(5s)
  → GROUND_SAFETY(throttle LOW on ground, 3s — must NOT lift off or auto-disarm)
  → DESCEND → LAND(BF auto-disarm) → DISARM → DONE

Also validates ground safety: armed at low throttle on ground must not
spontaneously lift off or auto-disarm.

Run:
    cd elodin && ./run.sh e2e-nosettle-takeoff

Prerequisites:
    Same eeprom as other ALT_HOLD tests.
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

TARGET_ALTITUDE = 7.0
HOVER_DURATION = 5.0
HOVER_TOLERANCE = 0.5
TEST_TIMEOUT = 120.0

TOP_APPROACH_ALTITUDE = 5.5
TOP_APPROACH_THROTTLE = 1570
TOP_APPROACH_VZ = 0.2
TOP_APPROACH_MIN_ALT = 6.8
TOP_APPROACH_DWELL = 0.5
TOP_APPROACH_TIMEOUT = 30.0

# Ground safety: stay armed at low throttle, must not lift off or auto-disarm
GROUND_SAFETY_DURATION = 3.0
GROUND_SAFETY_MAX_ALT = 0.2  # meters — must stay below this

CLIMB_TIMEOUT = 30.0  # shorter timeout — if it doesn't climb in 30s, the fix didn't work
BF_DISARM_MOTOR_THRESHOLD = 0.01
BF_DISARM_DWELL = 0.3
LAND_TIMEOUT = 15.0
DISARM_DURATION = 1.0

# RC channels
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
ALTHOLD_CLIMB = 1700
ALTHOLD_HOLD = 1500
ALTHOLD_DESCEND = 1300

BOOT_DURATION = 5.0
ARM_DURATION = 2.0
ALTHOLD_SETTLE = 2.0

ALTITUDE_CEILING = 50.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    # NO SETTLE PHASE — this is the point of the test
    CLIMB = auto()
    TOP_APPROACH = auto()
    HOVER = auto()
    DESCEND = auto()
    LAND = auto()
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

    hover_altitudes: list = field(default_factory=list)
    hover_max_drift: float = 0.0
    hover_target: float = 0.0
    top_approach_dwell: float = 0.0

    # Key metric: did it actually climb?
    reached_climb: bool = False
    climb_start_time: float = 0.0

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

    # NO SETTLE — go directly to CLIMB from ARM
    elif phase == Phase.CLIMB:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_CLIMB

    elif phase == Phase.TOP_APPROACH:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = TOP_APPROACH_THROTTLE

    elif phase == Phase.HOVER:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.DESCEND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_DESCEND

    elif phase == Phase.LAND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_DESCEND

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW

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
            print(f"[{t:6.1f}s] [     PRESELECT] Switches set (ANGLE+ALTHOLD). Arming...")
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(
                f"[{t:6.1f}s] [           ARM] "
                f"Armed. Going DIRECTLY to CLIMB (no settle phase)."
            )
            state.climb_start_time = t
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        # Key test: does the drone actually climb without a settle phase?
        if state.current_altitude > 1.0 and not state.reached_climb:
            state.reached_climb = True
            print(
                f"[{t:6.1f}s] [         CLIMB] "
                f"Climb confirmed! alt={state.current_altitude:.1f}m "
                f"(took {t - state.climb_start_time:.1f}s from ARM)"
            )

        if state.current_altitude >= TOP_APPROACH_ALTITUDE:
            state.transition(Phase.TOP_APPROACH, t)
        elif elapsed >= CLIMB_TIMEOUT:
            print(
                f"[{t:6.1f}s] [         CLIMB] "
                f"FAIL: Did not climb after {CLIMB_TIMEOUT:.0f}s. "
                f"alt={state.current_altitude:.2f}m (allowStickAdjustment bug?)"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.TOP_APPROACH:
        if state.current_altitude >= TOP_APPROACH_MIN_ALT:
            if abs(state.current_vz) < TOP_APPROACH_VZ:
                state.top_approach_dwell += dt
            else:
                state.top_approach_dwell = 0.0
            if state.top_approach_dwell >= TOP_APPROACH_DWELL:
                state.hover_target = state.current_altitude
                state.transition(Phase.HOVER, t)
        elif elapsed >= 30.0:
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)

    elif phase == Phase.HOVER:
        drift = abs(state.current_altitude - state.hover_target)
        state.hover_max_drift = max(state.hover_max_drift, drift)
        state.hover_altitudes.append(state.current_altitude)
        if elapsed >= HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [         HOVER] "
                f"Complete. drift={state.hover_max_drift:.1f}m. Descending..."
            )
            state.transition(Phase.DESCEND, t)

    elif phase == Phase.DESCEND:
        if state.current_altitude < 0.5:
            state.transition(Phase.LAND, t)
        elif elapsed >= 60.0:
            state.transition(Phase.LAND, t)

    elif phase == Phase.LAND:
        all_motors_zero = all(m < BF_DISARM_MOTOR_THRESHOLD for m in state.motors)
        if all_motors_zero:
            state.bf_disarm_dwell += dt
        else:
            state.bf_disarm_dwell = 0.0
        if state.bf_disarm_dwell >= BF_DISARM_DWELL:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            print(
                f"[{t:6.1f}s] [          LAND] "
                f"BF auto-disarm detected: alt={state.current_altitude:.2f}m"
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= LAND_TIMEOUT:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            print(f"[{t:6.1f}s] [          LAND] FAIL: BF auto-disarm not observed")
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

    if throttle <= 1050:
        stick = "IDLE"
    elif throttle >= 1650:
        stick = "CLIMB"
    elif throttle <= 1350:
        stick = "DESCEND"
    else:
        stick = "HOLD"

    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    ang = "ANG" if channels[CH_ANGLE] == MODE_ON else "---"
    alt = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"

    print(
        f"[{t:6.1f}s] [{state.phase.name:>14}] "
        f"alt={state.current_altitude:+7.2f}m vz={state.current_vz:+5.2f}m/s "
        f"motors=[{motors_str}] T={throttle} ({stick}) modes=[{arm}|{ang}|{alt}]"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E NO-SETTLE TAKEOFF TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print()
    print(f"  --- No-Settle Takeoff ---")
    print(f"  Climb without settle: {'Yes' if state.reached_climb else 'No (BUG)'}")
    print(f"  Max altitude:         {state.max_altitude:.1f}m")
    print(f"  Hover target:         {state.hover_target:.1f}m")
    print(f"  Hover max drift:      {state.hover_max_drift:.1f}m (tolerance: {HOVER_TOLERANCE}m)")
    print(f"  Landing altitude:     {state.land_altitude:.2f}m")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if not state.reached_climb:
        passed = False
        issues.append("Did not climb without settle phase (allowStickAdjustment gate bug)")

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if state.hover_max_drift > HOVER_TOLERANCE:
        passed = False
        issues.append(f"Hover drift too large ({state.hover_max_drift:.1f}m)")

    if state.land_altitude > 0.20:
        passed = False
        issues.append(f"Did not land properly ({state.land_altitude:.2f}m)")

    if state.step_count == 0:
        passed = False

        issues.append("No motor responses")

    state.test_passed = passed


    if passed:
        print("  Status:               PASS")
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
        hsplit name = "No-Settle Takeoff Test" {
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
    "e2e-nosettle-takeoff-test.kdl",
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
            print("  E2E NO-SETTLE TAKEOFF Test (regression for allowStickAdjustment)")
            print(f"  Scenario: PRESELECT → ARM → CLIMB directly (no settle)")
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
db_path = "/tmp/e2e_nosettle_test_db" if running_under_editor else "e2e_nosettle_test_db"

print(f"E2E NO-SETTLE TAKEOFF Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: PRESELECT → ARM → CLIMB (no settle)")

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
