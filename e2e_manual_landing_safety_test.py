#!/usr/bin/env python3
"""
E2E Manual-Landing Safety Test — anti-regression for Fix C.

Validates two scenarios for the manual-touchdown motor-stop predicate
(altHoldAllowsMotorStop's IN_PROGRESS branch):

  (1) LOW_PASS_NO_DISARM
      Drone descends through the 15-20cm motor-stop zone with descent stick
      for less than the MANUAL_TOUCHDOWN_DWELL_MS (1000ms) dwell. Pilot
      then pulls up and recovers. Drone MUST stay armed (motors must NOT
      drop to PWM 1000) and MUST climb back. Validates that brief stick
      dips during low passes do NOT false-positive disarm.

  (2) NEAR_GROUND_COMMIT_LAND
      Drone descends below 20cm and holds descent stick continuously for
      well over 1s. Predicate's dwell condition is satisfied → mixer cuts
      motors to PWM 1000 → drone touches down → strict landing detector
      naturally fires (real low vz now that motors are off) → BF auto-
      disarm. Validates the fix actually fires when intended.

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
    cd elodin && ./run.sh e2e-manual-landing-safety
    # or under realistic IGE:
    E2E_PHYSICS_PROFILE=realistic ./run.sh e2e-manual-landing-safety
"""

import os
import sys
import time
from collections import deque
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

# Throttle stick values
T_CLIMB = 1700       # well above center → climb at max manual climb rate
T_HOVER = 1500       # at center → hold altitude
T_DESCEND = 1300     # well below center but ABOVE mincheck (~1050)
                     # — same as nosettle test uses

# Phase durations
BOOT_DURATION = 5.0
ALTHOLD_SETTLE = 2.0
ARM_DURATION = 2.0
CLIMB_TIMEOUT = 30.0
HOVER_DURATION = 2.0

# Low-pass scenario
LOW_PASS_TARGET_ALT = 0.17        # m — clearly inside <0.20 motor-stop zone
LOW_PASS_DESCENT_TIMEOUT = 20.0   # max time to reach low-pass target
LOW_PASS_DWELL_S = 0.7            # seconds of descent stick BELOW 1s motor-stop dwell
LOW_PASS_RECOVER_TARGET = 0.50    # m — must climb back to here
LOW_PASS_RECOVER_TIMEOUT = 15.0

# Commit-land scenario
COMMIT_TARGET_ALT = 0.18          # m — clearly inside <0.20 motor-stop zone
COMMIT_DESCENT_TIMEOUT = 20.0
COMMIT_LAND_TIMEOUT = 10.0        # seconds to wait for BF auto-disarm after commit

# Pass criteria
MOTOR_OFF_PWM_THRESHOLD = 0.005   # normalized motor < this counts as "stopped"
LOW_PASS_DISARM_FAIL_THRESHOLD = 0.5  # if motors drop below MOTOR_OFF for > 0.5s during dwell, FAIL
COMMIT_FINAL_ALT_LIMIT = 0.20     # m — landed altitude must be below this

DISARM_DURATION = 1.0
TEST_TIMEOUT = 120.0

# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    CLIMB_TO_HOVER = auto()
    HOVER1 = auto()
    LOW_PASS_DESCEND = auto()      # descend toward LOW_PASS_TARGET_ALT
    LOW_PASS_DWELL = auto()        # hold descent stick for LOW_PASS_DWELL_S
    LOW_PASS_RECOVER = auto()      # climb back to LOW_PASS_RECOVER_TARGET
    HOVER2 = auto()
    COMMIT_DESCEND = auto()        # descend toward COMMIT_TARGET_ALT
    COMMIT_LAND = auto()           # hold descent stick, expect motor stop + disarm
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

    # Low-pass tracking
    low_pass_min_alt: float = 999.0
    low_pass_dwell_motor_off_seconds: float = 0.0  # time motors were ~off during dwell
    low_pass_recover_max_alt: float = 0.0
    low_pass_recovered: bool = False

    # Commit tracking
    commit_descent_started_t: float = 0.0
    commit_motor_stop_t: float = -1.0   # time motors first dropped to ~0
    commit_disarm_observed: bool = False
    commit_final_alt: float = 999.0
    commit_final_vz: float = 0.0

    last_print_time: float = -1.0
    results_printed: bool = False
    test_passed: bool = False

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>20}] --> [{new_phase.name}]")

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
        channels[CH_ALTHOLD] = MODE_ON
    elif phase == Phase.ARM:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
    elif phase == Phase.CLIMB_TO_HOVER:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = T_CLIMB
    elif phase == Phase.HOVER1 or phase == Phase.HOVER2:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = T_HOVER
    elif phase in (Phase.LOW_PASS_DESCEND, Phase.LOW_PASS_DWELL,
                   Phase.COMMIT_DESCEND, Phase.COMMIT_LAND):
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = T_DESCEND
    elif phase == Phase.LOW_PASS_RECOVER:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = T_CLIMB
    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW

    return channels


# ============================================================================
#  PHASE TRANSITIONS
# ============================================================================

def update_phase(state: TestState, t: float, dt: float):
    phase = state.phase
    elapsed = state.phase_elapsed(t)

    state.max_altitude = max(state.max_altitude, state.current_altitude)

    if phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.transition(Phase.PRESELECT, t)

    elif phase == Phase.PRESELECT:
        if elapsed >= ALTHOLD_SETTLE:
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            state.transition(Phase.CLIMB_TO_HOVER, t)

    elif phase == Phase.CLIMB_TO_HOVER:
        if state.current_altitude >= 0.65:
            state.transition(Phase.HOVER1, t)
        elif elapsed >= CLIMB_TIMEOUT:
            print(f"[{t:6.1f}s] FAIL: Climb timeout (alt={state.current_altitude:.2f}m)")
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOVER1:
        if elapsed >= HOVER_DURATION:
            state.transition(Phase.LOW_PASS_DESCEND, t)

    elif phase == Phase.LOW_PASS_DESCEND:
        state.low_pass_min_alt = min(state.low_pass_min_alt, state.current_altitude)
        if state.current_altitude <= LOW_PASS_TARGET_ALT:
            state.transition(Phase.LOW_PASS_DWELL, t)
        elif elapsed >= LOW_PASS_DESCENT_TIMEOUT:
            print(f"[{t:6.1f}s] FAIL: Low-pass descend timeout (alt={state.current_altitude:.2f}m)")
            state.transition(Phase.DISARM, t)

    elif phase == Phase.LOW_PASS_DWELL:
        # Track if motors went ~off during dwell — if they did sustainedly, that's a false positive disarm event
        max_motor = float(np.max(state.motors)) if state.motors is not None else 0.0
        if max_motor < MOTOR_OFF_PWM_THRESHOLD:
            state.low_pass_dwell_motor_off_seconds += dt
        if elapsed >= LOW_PASS_DWELL_S:
            state.transition(Phase.LOW_PASS_RECOVER, t)

    elif phase == Phase.LOW_PASS_RECOVER:
        state.low_pass_recover_max_alt = max(state.low_pass_recover_max_alt, state.current_altitude)
        if state.current_altitude >= LOW_PASS_RECOVER_TARGET:
            state.low_pass_recovered = True
            state.transition(Phase.HOVER2, t)
        elif elapsed >= LOW_PASS_RECOVER_TIMEOUT:
            print(f"[{t:6.1f}s] FAIL: Low-pass recovery timeout (max_alt={state.low_pass_recover_max_alt:.2f}m) "
                  f"— probably disarmed during low pass")
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOVER2:
        if elapsed >= HOVER_DURATION:
            state.transition(Phase.COMMIT_DESCEND, t)

    elif phase == Phase.COMMIT_DESCEND:
        if state.current_altitude <= COMMIT_TARGET_ALT:
            state.commit_descent_started_t = t
            state.transition(Phase.COMMIT_LAND, t)
        elif elapsed >= COMMIT_DESCENT_TIMEOUT:
            print(f"[{t:6.1f}s] FAIL: Commit descend timeout (alt={state.current_altitude:.2f}m)")
            state.transition(Phase.DISARM, t)

    elif phase == Phase.COMMIT_LAND:
        max_motor = float(np.max(state.motors)) if state.motors is not None else 0.0
        # Record first time motors actually went off
        if max_motor < MOTOR_OFF_PWM_THRESHOLD and state.commit_motor_stop_t < 0:
            state.commit_motor_stop_t = t
            print(f"[{t:6.1f}s] [          COMMIT_LAND] motors stopped at alt={state.current_altitude:.2f}m")
        # BF auto-disarm: detect by sustained motor off
        if max_motor < MOTOR_OFF_PWM_THRESHOLD and state.commit_motor_stop_t > 0:
            if t - state.commit_motor_stop_t >= 1.0:
                state.commit_disarm_observed = True
                state.commit_final_alt = state.current_altitude
                state.commit_final_vz = abs(state.current_vz)
                print(f"[{t:6.1f}s] [          COMMIT_LAND] BF auto-disarm detected: "
                      f"alt={state.current_altitude:.2f}m vz={abs(state.current_vz):.2f}m/s")
                state.transition(Phase.DISARM, t)
        elif elapsed >= COMMIT_LAND_TIMEOUT:
            state.commit_final_alt = state.current_altitude
            state.commit_final_vz = abs(state.current_vz)
            print(f"[{t:6.1f}s] [          COMMIT_LAND] FAIL: motor-stop / auto-disarm not observed")
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

    motors_str = ",".join(f"{m:.3f}" for m in state.motors) if state.motors is not None else "n/a"
    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]

    print(
        f"[{t:6.1f}s] [{state.phase.name:>20}] "
        f"alt={state.current_altitude:+6.2f}m vz={state.current_vz:+5.2f}m/s "
        f"motors=[{motors_str}] T={throttle}"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E MANUAL-LANDING SAFETY TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:                     {state.sim_time:.1f}s")
    print(f"  Lockstep steps:               {state.step_count}")
    print()
    print(f"  --- Low-Pass No-Disarm (motors must NOT stop during {LOW_PASS_DWELL_S}s dwell) ---")
    print(f"  Min altitude in low pass:     {state.low_pass_min_alt:.2f}m (target ~{LOW_PASS_TARGET_ALT}m)")
    print(f"  Motors-off seconds in dwell:  {state.low_pass_dwell_motor_off_seconds:.2f}s "
          f"(must be <{LOW_PASS_DISARM_FAIL_THRESHOLD}s)")
    print(f"  Recovery climb max alt:       {state.low_pass_recover_max_alt:.2f}m "
          f"(target {LOW_PASS_RECOVER_TARGET}m)")
    print(f"  Recovery completed:           {'Yes' if state.low_pass_recovered else 'No'}")
    print()
    print(f"  --- Near-Ground Commit Land (motors MUST stop and disarm fire) ---")
    if state.commit_motor_stop_t > 0:
        time_to_stop = state.commit_motor_stop_t - state.commit_descent_started_t
        print(f"  Motor-stop fired:             Yes (took {time_to_stop:.2f}s after entering <{COMMIT_TARGET_ALT}m)")
    else:
        print(f"  Motor-stop fired:             No (FAIL)")
    print(f"  Auto-disarm observed:         {'Yes' if state.commit_disarm_observed else 'No'}")
    print(f"  Final landing altitude:       {state.commit_final_alt:.2f}m (limit: {COMMIT_FINAL_ALT_LIMIT}m)")
    print(f"  Final landing velocity:       {state.commit_final_vz:.2f}m/s")
    print()

    passed = True
    issues = []

    # Low-pass criteria
    if state.low_pass_dwell_motor_off_seconds >= LOW_PASS_DISARM_FAIL_THRESHOLD:
        passed = False
        issues.append(
            f"FALSE-POSITIVE DISARM: motors went off for {state.low_pass_dwell_motor_off_seconds:.2f}s "
            f"during low-pass dwell (>{LOW_PASS_DISARM_FAIL_THRESHOLD}s threshold)")
    if not state.low_pass_recovered:
        passed = False
        issues.append(f"Low-pass recovery failed: max alt {state.low_pass_recover_max_alt:.2f}m "
                      f"< target {LOW_PASS_RECOVER_TARGET}m")

    # Commit-land criteria
    if state.commit_motor_stop_t < 0:
        passed = False
        issues.append("FIX-DOES-NOT-FIRE: motor-stop did not engage during commit-land")
    if not state.commit_disarm_observed:
        passed = False
        issues.append("Auto-disarm not observed during commit-land")
    if state.commit_final_alt > COMMIT_FINAL_ALT_LIMIT:
        passed = False
        issues.append(f"Did not land properly: {state.commit_final_alt:.2f}m > {COMMIT_FINAL_ALT_LIMIT}m")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses from Betaflight")

    state.test_passed = passed

    if passed:
        print("  Status:                       PASS")
    else:
        print("  Status:                       FAIL")
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
            print("  E2E MANUAL-LANDING SAFETY Test")
            print("  Validates Fix C anti-regression: low-pass no-disarm + commit-land")
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
            pass  # Single-tick timeouts are normal during BF task scheduling jitter

    print_status(s, t)

    if s.phase == Phase.DONE and not s.results_printed:
        b.stop()
        elapsed = time.time() - _start_time[0]
        print(f"\nSimulation: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)")
        print_results(s)

    if tick >= max_ticks - 1 and not s.results_printed:
        print(f"\n[{t:6.1f}s] TEST TIMEOUT")
        b.stop()
        print_results(s)


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv
db_path = "/tmp/e2e_manual_landing_safety_test_db" if running_under_editor else "e2e_manual_landing_safety_test_db"

print(f"E2E MANUAL-LANDING SAFETY Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")

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
