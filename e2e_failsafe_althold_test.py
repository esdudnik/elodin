#!/usr/bin/env python3
"""
E2E Failsafe Landing Test: Takeoff, hover, trigger failsafe, validate BF-owned descent + disarm.

Tests Betaflight's failsafe landing path with ALT_HOLD:
- Normal takeoff and climb to hover altitude
- Trigger BOXFAILSAFE switch → BF enters FAILSAFE_LANDING (stage 2)
- BF descends using its own velocity ramp (alt_hold_multirotor.c)
- Shared landing detector (isAltHoldLandingDetected) fires → FAILSAFE_LANDED → disarm
- Validates: disarm happens well before failsafe timeout (60s)

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE+ALTHOLD, 2s) → ARM(2s) → SETTLE(1s)
  → CLIMB(to 5.5m) → TOP_APPROACH(to 6.8m) → HOVER(5s)
  → TRIGGER_FAILSAFE(flip AUX4) → FAILSAFE_DESCENT(monitor until BF disarms)
  → DONE

Run:
    cd elodin && ./run.sh e2e-failsafe-althold

Prerequisites:
    1. Build Betaflight SITL: cd betaflight && make TARGET=SITL
    2. Configure eeprom:
        set acc_calibration = 0,0,0,1
        set failsafe_delay = 200
        set ap_hover_throttle = 1130
        set d_pitch = 5
        set d_roll = 5
        set failsafe_switch_mode = STAGE2
        set failsafe_procedure = AUTO-LAND
        aux 0 0 0 1700 2100 0 0
        aux 1 1 1 1700 2100 0 0
        aux 2 3 2 1700 2100 0 0
        aux 3 27 3 1700 2100 0 0
        save
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
    print("Build it: cd betaflight && make arm_sdk_install && make TARGET=SITL")
    sys.exit(1)


# ============================================================================
#  TEST PARAMETERS
# ============================================================================

TARGET_ALTITUDE = 7.0
HOVER_DURATION = 5.0         # shorter hover — just enough to stabilize before failsafe
HOVER_TOLERANCE = 0.5
LAND_MAX_VELOCITY = 1.5
TEST_TIMEOUT = 120.0         # enough for climb + hover + failsafe descent

# Top approach
TOP_APPROACH_ALTITUDE = 5.5
TOP_APPROACH_THROTTLE = 1570
TOP_APPROACH_VZ = 0.2
TOP_APPROACH_MIN_ALT = 6.8
TOP_APPROACH_DWELL = 0.5
TOP_APPROACH_TIMEOUT = 30.0
HOVER_TARGET_TOLERANCE = 2.0

# Failsafe descent
FAILSAFE_DESCENT_TIMEOUT = 30.0  # seconds — must land before this (failsafe timeout is 60s)

# BF auto-disarm detection
BF_DISARM_MOTOR_THRESHOLD = 0.01
BF_DISARM_DWELL = 0.3

# RC channels
CH_ROLL = 0
CH_PITCH = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_ARM = 4
CH_ANGLE = 5
CH_ALTHOLD = 6
CH_FAILSAFE = 7   # AUX4 — mapped to BOXFAILSAFE (permanentId=27)

RC_CENTER = 1500
RC_LOW = 1000
MODE_ON = 1800
MODE_OFF = 1000
ALTHOLD_CLIMB = 1700
ALTHOLD_HOLD = 1500

# Phase durations
BOOT_DURATION = 5.0
ARM_DURATION = 2.0
ALTHOLD_SETTLE = 2.0
DISARM_DURATION = 1.0
CLIMB_TIMEOUT = 90.0

# Crash detection
CRASH_MOTOR_ASYMMETRY = 0.7
CRASH_GROUND_STUCK_TIME = 3.0
CRASH_MAX_VELOCITY = 8.0
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
    TOP_APPROACH = auto()
    HOVER = auto()
    TRIGGER_FAILSAFE = auto()   # flip BOXFAILSAFE switch
    FAILSAFE_DESCENT = auto()   # monitor BF-owned failsafe descent
    DISARM = auto()
    DONE = auto()


@dataclass
class TestState:
    phase: Phase = Phase.BOOT
    phase_start_time: float = 0.0
    sim_time: float = 0.0
    tick: int = 0

    # Motor data
    motors: np.ndarray = field(default_factory=lambda: np.zeros(4))
    max_motor: float = 0.0
    step_count: int = 0

    # Flight data
    current_altitude: float = 0.0
    current_vz: float = 0.0
    max_altitude: float = 0.0
    initial_altitude: float = 0.0
    baro_altitude: float = 0.0

    # Hover tracking
    hover_altitudes: list = field(default_factory=list)
    hover_max_drift: float = 0.0
    hover_target: float = 0.0
    top_approach_dwell: float = 0.0

    # Landing tracking
    land_altitude: float = 0.0
    land_velocity: float = 0.0
    bf_disarm_dwell: float = 0.0

    # Failsafe tracking
    failsafe_trigger_alt: float = 0.0    # altitude when failsafe was triggered
    failsafe_descent_started: bool = False  # descent confirmed after trigger
    failsafe_descent_time: float = 0.0   # time from trigger to disarm

    # Crash detection
    crash_detected: bool = False
    crash_reason: str = ""
    ground_stuck_time: float = 0.0

    # Diagnostics
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

    elif phase == Phase.TRIGGER_FAILSAFE:
        # Flip BOXFAILSAFE while maintaining hover RC state.
        # With failsafe_switch_mode=STAGE2, BF enters FAILSAFE_LANDING immediately.
        # BF overrides RC during failsafe, but we keep sending for lockstep.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_FAILSAFE] = MODE_ON

    elif phase == Phase.FAILSAFE_DESCENT:
        # Keep BOXFAILSAFE on. BF owns the descent entirely.
        # ARM still commanded — BF auto-disarm via FAILSAFE_LANDED path.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_FAILSAFE] = MODE_ON

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW
        channels[CH_FAILSAFE] = MODE_OFF

    return channels


def check_crash(state: TestState, t: float, dt: float) -> bool:
    if state.phase in (Phase.BOOT, Phase.PRESELECT, Phase.ARM, Phase.SETTLE, Phase.DISARM, Phase.DONE):
        return False

    motors = state.motors
    if np.max(motors) < 0.02:
        return False

    if state.current_altitude > ALTITUDE_CEILING:
        state.crash_detected = True
        state.crash_reason = f"Altitude ceiling exceeded ({state.current_altitude:.1f}m > {ALTITUDE_CEILING}m)"
        return True

    if abs(state.current_vz) > CRASH_MAX_VELOCITY:
        state.crash_detected = True
        state.crash_reason = f"Excessive vertical speed ({state.current_vz:.1f}m/s)"
        return True

    return False


def update_phase(state: TestState, t: float, dt: float):
    phase = state.phase
    elapsed = state.phase_elapsed(t)

    if check_crash(state, t, dt):
        print(f"[{t:6.1f}s] CRASH DETECTED: {state.crash_reason}")
        state.land_altitude = state.current_altitude
        state.land_velocity = abs(state.current_vz)
        state.transition(Phase.DISARM, t)
        return

    if phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.initial_altitude = state.current_altitude
            state.transition(Phase.PRESELECT, t)

    elif phase == Phase.PRESELECT:
        if elapsed >= ALTHOLD_SETTLE:
            print(
                f"[{t:6.1f}s] [     PRESELECT] "
                f"Switches set (ANGLE+ALTHOLD). Arming..."
            )
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(
                f"[{t:6.1f}s] [           ARM] "
                f"Armed. ALTHOLD active. Reference altitude: {state.current_altitude:.1f}m"
            )
            state.transition(Phase.SETTLE, t)

    elif phase == Phase.SETTLE:
        if elapsed >= 1.0:
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        if state.current_altitude >= TOP_APPROACH_ALTITUDE:
            state.transition(Phase.TOP_APPROACH, t)
        elif elapsed >= CLIMB_TIMEOUT:
            print(f"[{t:6.1f}s] CLIMB TIMEOUT at {state.current_altitude:.1f}m")
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
        elif elapsed >= TOP_APPROACH_TIMEOUT:
            print(f"[{t:6.1f}s] TOP_APPROACH TIMEOUT at {state.current_altitude:.1f}m")
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)

    elif phase == Phase.HOVER:
        drift = abs(state.current_altitude - state.hover_target)
        state.hover_max_drift = max(state.hover_max_drift, drift)
        state.hover_altitudes.append(state.current_altitude)

        if elapsed >= HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [         HOVER] "
                f"Complete ({HOVER_DURATION:.0f}s). Triggering failsafe..."
            )
            state.transition(Phase.TRIGGER_FAILSAFE, t)

    elif phase == Phase.TRIGGER_FAILSAFE:
        # Immediate transition — BOXFAILSAFE switch is set in RC channels.
        # BF processes it on next tick: failsafe.c:301 forces FAILSAFE_LANDING.
        state.failsafe_trigger_alt = state.current_altitude
        print(
            f"[{t:6.1f}s] [TRIGGER_FAILSAFE] "
            f"BOXFAILSAFE ON at alt={state.current_altitude:.2f}m. "
            f"Monitoring BF failsafe descent..."
        )
        state.transition(Phase.FAILSAFE_DESCENT, t)

    elif phase == Phase.FAILSAFE_DESCENT:
        # Monitor BF-owned failsafe descent.
        # Success: BF auto-disarm detected (motors=0 while ARM+FAILSAFE still commanded).
        # Fail: timeout before BF disarms.

        # Track that descent actually began
        if not state.failsafe_descent_started and state.current_vz < -0.3:
            state.failsafe_descent_started = True
            print(
                f"[{t:6.1f}s] [FAILSAFE_DESCENT] "
                f"Descent confirmed: vz={state.current_vz:.2f}m/s alt={state.current_altitude:.2f}m"
            )

        # Detect BF auto-disarm (FAILSAFE_LANDED → disarm)
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
                f"BF failsafe disarm detected: alt={state.current_altitude:.2f}m "
                f"vz={state.current_vz:.2f}m/s "
                f"descent_time={elapsed:.1f}s (from {state.failsafe_trigger_alt:.1f}m)"
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= FAILSAFE_DESCENT_TIMEOUT:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            state.failsafe_descent_time = elapsed
            print(
                f"[{t:6.1f}s] [FAILSAFE_DESCENT] "
                f"FAIL: BF failsafe disarm not observed after {FAILSAFE_DESCENT_TIMEOUT:.0f}s. "
                f"alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s "
                f"motors={[f'{m:.3f}' for m in state.motors]}"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.DISARM:
        if elapsed >= DISARM_DURATION:
            state.transition(Phase.DONE, t)


def print_status(state: TestState, t: float):
    if int(t) <= state.last_print_time:
        return
    state.last_print_time = int(t)

    phase_name = state.phase.name
    motors_str = ",".join(f"{m:.3f}" for m in state.motors)
    baro_str = f"baro={state.baro_altitude:+7.2f}m"

    # Throttle description
    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]
    failsafe_on = channels[CH_FAILSAFE] == MODE_ON
    if failsafe_on:
        stick_desc = f"T={throttle} (FAILSAFE)"
    elif throttle <= 1050:
        stick_desc = f"T={throttle} (IDLE)"
    elif throttle >= 1650:
        stick_desc = f"T={throttle} (CLIMB)"
    elif throttle <= 1350:
        stick_desc = f"T={throttle} (DESCEND)"
    else:
        stick_desc = f"T={throttle} (HOLD)"

    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    ang = "ANG" if channels[CH_ANGLE] == MODE_ON else "---"
    alt = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"
    fs = " FS" if failsafe_on else ""

    print(
        f"[{t:6.1f}s] [{phase_name:>14}] "
        f"alt={state.current_altitude:+7.2f}m "
        f"vz={state.current_vz:+6.2f}m/s "
        f"{baro_str} "
        f"motors=[{motors_str}] "
        f"{stick_desc} "
        f"modes=[{arm}|{ang}|{alt}]{fs}"
    )


def print_results(state: TestState):
    print()
    print("=" * 70)
    print("  E2E FAILSAFE LANDING TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print(f"  Max motor value:      {state.max_motor:.3f}")
    print()
    print(f"  Target altitude:      {TARGET_ALTITUDE:.0f}m")
    print(f"  Max altitude:         {state.max_altitude:.1f}m")
    print(f"  Hover target:         {state.hover_target:.1f}m")
    print(f"  Hover max drift:      {state.hover_max_drift:.1f}m (tolerance: {HOVER_TOLERANCE}m)")
    if state.hover_altitudes:
        avg = sum(state.hover_altitudes) / len(state.hover_altitudes)
        print(f"  Hover avg altitude:   {avg:.1f}m")
    print()
    print(f"  --- Failsafe Landing ---")
    print(f"  Trigger altitude:     {state.failsafe_trigger_alt:.1f}m")
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
        issues.append("Failsafe descent never started (vz never went below -0.3m/s)")

    if state.failsafe_descent_time >= FAILSAFE_DESCENT_TIMEOUT:
        passed = False
        issues.append(
            f"BF failsafe disarm not observed within {FAILSAFE_DESCENT_TIMEOUT:.0f}s "
            f"(landing detector did not trigger early exit)"
        )

    if state.land_altitude > 0.20:
        passed = False
        issues.append(f"Did not land properly ({state.land_altitude:.2f}m > 0.20m)")

    if state.hover_max_drift > HOVER_TOLERANCE:
        passed = False
        issues.append(f"Hover drift too large ({state.hover_max_drift:.1f}m > {HOVER_TOLERANCE}m)")

    if state.step_count == 0:
        passed = False

        issues.append("No motor responses from Betaflight")

    if state.max_motor < 0.02:
        passed = False
        issues.append("Motors never spun up")

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
        hsplit name = "Failsafe Test" {
            viewport name=Viewport pos="drone.world_pos.translate_world(5.0, 5.0, 3.0)" look_at="drone.world_pos" show_grid=#true active=#true
            vsplit share=0.3 {
                graph "drone.motor_command" name="Motor Commands"
                graph "drone.motor_thrust" name="Motor Thrust"
            }
            vsplit share=0.3 {
                graph "drone.world_pos.linear()" name="Position (ENU)"
                graph "drone.world_vel.linear()" name="Velocity"
            }
        }
    }
    object_3d drone.world_pos {
        glb path="edu-450-v2-drone.glb" rotate="(0.0, 0.0, 0.0)" translate="(0.0, 1.0, 0.0)" scale=10.0
    }
    """,
    "e2e-failsafe-test.kdl",
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
            print("  E2E FAILSAFE LANDING Test")
            print(f"  Scenario: climb to {TARGET_ALTITUDE}m, hover {HOVER_DURATION}s, "
                  f"trigger BOXFAILSAFE, validate BF-owned descent + disarm")
            print(f"  Config: {config.mass}kg quad, ANGLE + ALT_HOLD + BOXFAILSAFE")
            print(f"  Timeout: {TEST_TIMEOUT}s")
            print("=" * 70)
            print()

            bridge_obj.start()
            _bridge[0] = bridge_obj
            print("[  0.0s] [          INIT] Waiting 2s for Betaflight initialization...")
        except Exception as e:
            print(f"[INIT] ERROR: Failed to initialize bridge: {e}")
            _bridge[0] = None
            return
        time.sleep(2)

        # Warmup
        print("[  0.0s] [          INIT] Sending warmup packets...")
        warmup_buf = SensorDataBuffer()
        warmup_fdm = warmup_buf.build_fdm()
        warmup_channels = np.full(MAX_RC_CHANNELS, RC_CENTER, dtype=np.uint16)
        warmup_channels[CH_THROTTLE] = RC_LOW
        warmup_channels[CH_ARM] = MODE_OFF
        warmup_rc = RCPacket(timestamp=0.0, channels=warmup_channels)

        warmup_ok = 0
        warmup_total = int(0.5 / config.sim_time_step)
        for i in range(warmup_total):
            warmup_fdm.timestamp = i * config.sim_time_step
            warmup_rc.timestamp = i * config.sim_time_step
            try:
                _bridge[0].step(warmup_fdm, warmup_rc)
                warmup_ok += 1
            except TimeoutError:
                pass

        print(f"[  0.0s] [          INIT] Warmup: {warmup_ok}/{warmup_total} responses")
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

    # Read sensor data
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
            print(f"[{t:6.1f}s] WARNING: Could not read sensor data: {e}")
        buf.timestamp = t

    # Phase transitions
    update_phase(s, t, dt=config.sim_time_step)

    # Build RC channels
    channels = build_rc_channels(s)

    # Lockstep
    fdm = buf.build_fdm()
    rc = RCPacket(timestamp=t, channels=channels)

    try:
        s.motors = b.step(fdm, rc)
        s.max_motor = max(s.max_motor, float(np.max(s.motors)))
        s.step_count += 1
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        if s.phase not in (Phase.BOOT, Phase.DONE):
            print(f"[{t:6.1f}s] [{s.phase.name:>14}] WARNING: Motor response timeout")

    # Status
    print_status(s, t)

    # Completion
    if s.phase == Phase.DONE and not s.results_printed:
        b.stop()
        elapsed = time.time() - _start_time[0]
        print(f"\nSimulation: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)")
        print_results(s)
        s.results_printed = True

    # Timeout
    if tick >= max_ticks - 1 and not s.results_printed:
        print(f"\n[{t:6.1f}s] TEST TIMEOUT ({TEST_TIMEOUT}s) in phase {s.phase.name}")
        s.land_altitude = s.current_altitude
        s.land_velocity = abs(s.current_vz)
        b.stop()
        print_results(s)
        s.results_printed = True


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv

if running_under_editor:
    db_path = "/tmp/e2e_failsafe_test_db"
else:
    db_path = "e2e_failsafe_test_db"
use_interactive = False

print(f"E2E FAILSAFE LANDING Test")
print(f"  SITL binary: {BETAFLIGHT_PATH.name}")
print(f"  Sim rate: {1.0/config.sim_time_step:.0f}Hz, timeout: {TEST_TIMEOUT}s")
print(f"  Scenario: takeoff -> hover -> trigger BOXFAILSAFE -> BF-owned descent -> disarm")
if running_under_editor:
    print(f"  Mode: editor (db at {db_path})")
else:
    print(f"  Mode: headless (db at {db_path})")

world.run(
    system,
    sim_time_step=config.sim_time_step,
    run_time_step=config.sim_time_step,
    max_ticks=max_ticks,
    post_step=e2e_post_step,
    db_path=db_path,
    interactive=use_interactive,
    backend="jax",
)

# Exit with test result code
sys.exit(0 if _state[0] and _state[0].test_passed else 1)
