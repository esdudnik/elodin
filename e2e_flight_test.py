#!/usr/bin/env python3
"""
E2E Horizontal Flight Test: Takeoff, climb, fly left/right, hover near ground, land.

Tests Betaflight's ALT_HOLD + ANGLE mode during horizontal flight:
- Maintains altitude while commanding roll (lateral movement)
- Stabilizes after stick release (velocity decay)
- Full flight cycle: takeoff, horizontal maneuvers, descent, near-ground hover, landing

Flight phases:
  BOOT(5s) -> PRESELECT(ANGLE+ALTHOLD on, ARM off, 2s) -> ARM(2s) -> SETTLE(1s)
  -> CLIMB(to 5.5m) -> TOP_APPROACH(to 6.8m) -> HOVER(5s)
  -> FLY_LEFT(5s) -> STABILIZE(3s) -> FLY_RIGHT(5s) -> STABILIZE(3s)
  -> DESCEND(to 2m) -> RAMP_APPROACH(to 0.5m) -> LOW_HOVER(10s)
  -> LAND(contact-based) -> DISARM

Run:
    cd elodin && ./run.sh e2e-flight          # headless
    cd elodin && ./run.sh e2e-flight-editor   # with 3D viewport

Prerequisites:
    1. Build Betaflight SITL: cd betaflight && make TARGET=SITL
    2. Configure eeprom (fresh reset):
        set acc_calibration = 0,0,0,1
        set failsafe_delay = 200
        set ap_hover_throttle = 1130
        set d_pitch = 5
        set d_roll = 5
        aux 0 0 0 1700 2100 0 0
        aux 1 1 1 1700 2100 0 0
        aux 2 3 2 1700 2100 0 0
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
HOVER_DURATION = 5.0         # shorter hover before flight maneuvers
HOVER_TOLERANCE = 0.5
TEST_TIMEOUT = 200.0

# Top approach
TOP_APPROACH_ALTITUDE = 5.5
TOP_APPROACH_THROTTLE = 1570
TOP_APPROACH_VZ = 0.2
TOP_APPROACH_MIN_ALT = 6.8
TOP_APPROACH_DWELL = 0.5
TOP_APPROACH_TIMEOUT = 30.0
HOVER_TARGET_TOLERANCE = 2.0

# Horizontal flight
FLY_DURATION = 5.0           # seconds per direction
FLY_ROLL_LEFT = 1400         # gentle left bank (~10 degrees in ANGLE mode)
FLY_ROLL_RIGHT = 1600        # gentle right bank
STABILIZE_DURATION = 3.0     # seconds to settle after stick release
STABILIZE_VXY_THRESHOLD = 0.5  # m/s — horizontal speed must decay below this
FLIGHT_ALT_TOLERANCE = 0.8   # meters — altitude hold tolerance during horizontal flight
MIN_HORIZONTAL_DISTANCE = 1.0  # meters — must move at least this far in commanded direction

# Descent & landing (reuse althold parameters)
APPROACH_ALTITUDE = 2.0
APPROACH_THROTTLE = 1380
RAMP_THROTTLE_HIGH = 1450
RAMP_APPROACH_TIMEOUT = 60.0
LOW_HOVER_ENTRY_ALT = 0.5
LOW_HOVER_ENTRY_VZ = 0.2
LOW_HOVER_DWELL = 1.0
LOW_HOVER_DURATION = 10.0
LOW_HOVER_DRIFT_TOLERANCE = 0.5
# BF auto-disarm detection (Phase 3: BF owns landing)
BF_DISARM_MOTOR_THRESHOLD = 0.01  # all motors must be below this
BF_DISARM_DWELL = 0.3             # seconds — motors must stay zero for this long
LAND_TIMEOUT = 15.0
DESCEND_TIMEOUT = 60.0

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
RUNAWAY_CLIMB_TIME = 2.0


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
    FLY_LEFT = auto()
    STABILIZE_1 = auto()      # after FLY_LEFT
    FLY_RIGHT = auto()
    STABILIZE_2 = auto()      # after FLY_RIGHT
    DESCEND = auto()
    RAMP_APPROACH = auto()
    LOW_HOVER = auto()
    LAND = auto()
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
    current_x: float = 0.0
    current_y: float = 0.0
    current_vx: float = 0.0
    current_vy: float = 0.0
    max_altitude: float = 0.0
    initial_altitude: float = 0.0
    baro_altitude: float = 0.0

    # Hover tracking
    hover_altitudes: list = field(default_factory=list)
    hover_max_drift: float = 0.0
    hover_target: float = 0.0
    top_approach_dwell: float = 0.0
    low_hover_dwell: float = 0.0
    low_hover_target: float = 0.0
    low_hover_max_drift: float = 0.0

    # Horizontal flight tracking
    fly_left_start_x: float = 0.0
    fly_left_start_y: float = 0.0
    fly_left_distance: float = 0.0
    fly_right_start_x: float = 0.0
    fly_right_start_y: float = 0.0
    fly_right_distance: float = 0.0
    flight_max_alt_error: float = 0.0  # max altitude deviation during horizontal flight
    flight_hover_alt: float = 0.0      # altitude at start of horizontal flight
    stabilize_final_vxy: float = 0.0   # horizontal speed at end of last stabilize

    # Landing tracking
    land_altitude: float = 0.0
    land_velocity: float = 0.0
    ground_contact_time: float = 0.0
    bf_disarm_dwell: float = 0.0

    # Crash detection
    crash_detected: bool = False
    crash_reason: str = ""
    ground_stuck_time: float = 0.0
    runaway_climb_time: float = 0.0

    # Diagnostics
    last_print_time: float = -1.0
    results_printed: bool = False
    quat_xyzw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    gyro_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>14}] --> [{new_phase.name}]")

    def phase_elapsed(self, t: float) -> float:
        return t - self.phase_start_time

    def horizontal_speed(self) -> float:
        return (self.current_vx**2 + self.current_vy**2) ** 0.5


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
        # Switch preselection: ANGLE+ALTHOLD on, ARM off.
        # BF sees switch states but ALT_HOLD_MODE does not activate (gated by ARMED).
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

    elif phase == Phase.FLY_LEFT:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_ROLL] = FLY_ROLL_LEFT

    elif phase == Phase.STABILIZE_1:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.FLY_RIGHT:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_ROLL] = FLY_ROLL_RIGHT

    elif phase == Phase.STABILIZE_2:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.DESCEND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_DESCEND

    elif phase == Phase.RAMP_APPROACH:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        alt = max(state.current_altitude, 0.0)
        progress = (APPROACH_ALTITUDE - alt) / (APPROACH_ALTITUDE - LOW_HOVER_ENTRY_ALT)
        progress = max(0.0, min(1.0, progress))
        throttle = int(APPROACH_THROTTLE + progress * (RAMP_THROTTLE_HIGH - APPROACH_THROTTLE))
        channels[CH_THROTTLE] = throttle

    elif phase == Phase.LOW_HOVER:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.LAND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = APPROACH_THROTTLE

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.DONE:
        channels[CH_THROTTLE] = RC_LOW

    return channels


# ============================================================================
#  CRASH DETECTION
# ============================================================================

def check_crash(state: TestState, t: float, dt: float) -> bool:
    if state.phase in (Phase.BOOT, Phase.PRESELECT, Phase.ARM, Phase.SETTLE, Phase.DISARM, Phase.DONE):
        return False

    motors = state.motors
    if np.max(motors) < 0.02:
        return False

    motor_asymmetry = float(np.max(motors) - np.min(motors))
    on_ground = state.current_altitude < 0.5

    if on_ground and motor_asymmetry > CRASH_MOTOR_ASYMMETRY:
        state.ground_stuck_time += dt
        if state.ground_stuck_time >= CRASH_GROUND_STUCK_TIME:
            state.crash_detected = True
            state.crash_reason = f"Drone flipped on ground. Motors asymmetry: {motor_asymmetry:.2f}"
            return True
    else:
        state.ground_stuck_time = 0.0

    if abs(state.current_vz) > CRASH_MAX_VELOCITY:
        state.crash_detected = True
        state.crash_reason = f"Extreme vertical velocity: vz={state.current_vz:+.1f}m/s"
        return True

    if state.current_altitude > ALTITUDE_CEILING:
        state.crash_detected = True
        state.crash_reason = f"Altitude ceiling exceeded: alt={state.current_altitude:.1f}m"
        return True

    descent_phases = (Phase.DESCEND, Phase.RAMP_APPROACH, Phase.LOW_HOVER, Phase.LAND)
    if state.phase in descent_phases:
        if state.current_vz > 0.5:
            state.runaway_climb_time += dt
        else:
            state.runaway_climb_time = 0.0
        if state.runaway_climb_time >= RUNAWAY_CLIMB_TIME:
            state.crash_detected = True
            state.crash_reason = (
                f"Runaway climb during {state.phase.name}: "
                f"vz={state.current_vz:+.1f}m/s sustained for {state.runaway_climb_time:.1f}s. "
                f"alt={state.current_altitude:.1f}m"
            )
            return True
    else:
        state.runaway_climb_time = 0.0

    return False


# ============================================================================
#  PHASE TRANSITIONS
# ============================================================================

def update_phase(state: TestState, t: float, dt: float = 0.001):
    phase = state.phase
    elapsed = state.phase_elapsed(t)

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
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)

    elif phase == Phase.TOP_APPROACH:
        vz_abs = abs(state.current_vz)
        alt_high_enough = state.current_altitude >= TOP_APPROACH_MIN_ALT
        if alt_high_enough and vz_abs < TOP_APPROACH_VZ:
            state.top_approach_dwell += dt
        else:
            state.top_approach_dwell = 0.0
        if state.top_approach_dwell >= TOP_APPROACH_DWELL:
            print(
                f"[{t:6.1f}s] [  TOP_APPROACH] "
                f"At target: alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s"
            )
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)
        elif elapsed >= TOP_APPROACH_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"TOP_APPROACH timeout: alt={state.current_altitude:.1f}m"
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOVER:
        drift = abs(state.current_altitude - state.hover_target)
        state.hover_altitudes.append(state.current_altitude)
        state.hover_max_drift = max(state.hover_max_drift, drift)
        if elapsed >= HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [         HOVER] "
                f"Complete ({HOVER_DURATION:.0f}s). Drift: {state.hover_max_drift:.1f}m"
            )
            state.flight_hover_alt = state.current_altitude
            state.fly_left_start_x = state.current_x
            state.fly_left_start_y = state.current_y
            state.transition(Phase.FLY_LEFT, t)

    elif phase == Phase.FLY_LEFT:
        # Track altitude error during flight
        alt_error = abs(state.current_altitude - state.flight_hover_alt)
        state.flight_max_alt_error = max(state.flight_max_alt_error, alt_error)
        if elapsed >= FLY_DURATION:
            dx = state.current_x - state.fly_left_start_x
            dy = state.current_y - state.fly_left_start_y
            state.fly_left_distance = (dx**2 + dy**2) ** 0.5
            print(
                f"[{t:6.1f}s] [      FLY_LEFT] "
                f"Complete. Distance: {state.fly_left_distance:.1f}m "
                f"alt_error: {state.flight_max_alt_error:.2f}m"
            )
            state.transition(Phase.STABILIZE_1, t)

    elif phase == Phase.STABILIZE_1:
        vxy = state.horizontal_speed()
        if elapsed >= STABILIZE_DURATION:
            print(
                f"[{t:6.1f}s] [   STABILIZE_1] "
                f"Complete. vxy={vxy:.2f}m/s"
            )
            state.fly_right_start_x = state.current_x
            state.fly_right_start_y = state.current_y
            state.transition(Phase.FLY_RIGHT, t)

    elif phase == Phase.FLY_RIGHT:
        alt_error = abs(state.current_altitude - state.flight_hover_alt)
        state.flight_max_alt_error = max(state.flight_max_alt_error, alt_error)
        if elapsed >= FLY_DURATION:
            dx = state.current_x - state.fly_right_start_x
            dy = state.current_y - state.fly_right_start_y
            state.fly_right_distance = (dx**2 + dy**2) ** 0.5
            print(
                f"[{t:6.1f}s] [     FLY_RIGHT] "
                f"Complete. Distance: {state.fly_right_distance:.1f}m "
                f"alt_error: {state.flight_max_alt_error:.2f}m"
            )
            state.transition(Phase.STABILIZE_2, t)

    elif phase == Phase.STABILIZE_2:
        vxy = state.horizontal_speed()
        if elapsed >= STABILIZE_DURATION:
            state.stabilize_final_vxy = vxy
            print(
                f"[{t:6.1f}s] [   STABILIZE_2] "
                f"Complete. vxy={vxy:.2f}m/s"
            )
            state.transition(Phase.DESCEND, t)

    elif phase == Phase.DESCEND:
        if state.current_altitude <= APPROACH_ALTITUDE:
            state.transition(Phase.RAMP_APPROACH, t)
        elif elapsed >= DESCEND_TIMEOUT:
            state.transition(Phase.RAMP_APPROACH, t)

    elif phase == Phase.RAMP_APPROACH:
        vz_abs = abs(state.current_vz)
        if state.current_altitude <= LOW_HOVER_ENTRY_ALT and vz_abs < LOW_HOVER_ENTRY_VZ:
            state.low_hover_dwell += dt
        else:
            state.low_hover_dwell = 0.0
        if state.low_hover_dwell >= LOW_HOVER_DWELL:
            print(
                f"[{t:6.1f}s] [ RAMP_APPROACH] "
                f"Low & slow: alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s"
            )
            state.transition(Phase.LOW_HOVER, t)
        elif elapsed >= RAMP_APPROACH_TIMEOUT:
            state.transition(Phase.LAND, t)

    elif phase == Phase.LOW_HOVER:
        if state.low_hover_target == 0.0:
            state.low_hover_target = state.current_altitude
        drift = abs(state.current_altitude - state.low_hover_target)
        state.low_hover_max_drift = max(state.low_hover_max_drift, drift)
        if elapsed >= LOW_HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [     LOW_HOVER] "
                f"Complete. alt={state.current_altitude:.2f}m drift={state.low_hover_max_drift:.2f}m"
            )
            state.transition(Phase.LAND, t)

    elif phase == Phase.LAND:
        # BF auto-disarm detection: ARM stays commanded, wait for BF to drop motors.
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
                f"BF auto-disarm detected: alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s "
                f"motors=0 for {state.bf_disarm_dwell:.1f}s (ARM still commanded)"
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= LAND_TIMEOUT:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            print(
                f"[{t:6.1f}s] [          LAND] "
                f"FAIL: BF auto-disarm not observed after {LAND_TIMEOUT:.0f}s. "
                f"alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s "
                f"motors={[f'{m:.3f}' for m in state.motors]}"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.DISARM:
        if elapsed >= DISARM_DURATION:
            state.transition(Phase.DONE, t)


# ============================================================================
#  STATUS PRINTING
# ============================================================================

def print_status(state: TestState, t: float):
    if t - state.last_print_time < 1.0:
        return
    state.last_print_time = t

    phase_name = state.phase.name
    motors_str = ",".join(f"{m:.3f}" for m in state.motors)

    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]
    roll = channels[CH_ROLL]

    stick_desc = f"T={throttle}"
    if roll != RC_CENTER:
        stick_desc += f" R={roll}"

    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    ang = "ANG" if channels[CH_ANGLE] == MODE_ON else "---"
    alt = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"

    vxy = state.horizontal_speed()

    print(
        f"[{t:6.1f}s] [{phase_name:>14}] "
        f"alt={state.current_altitude:+7.2f}m vz={state.current_vz:+5.2f}m/s "
        f"xy=({state.current_x:+.1f},{state.current_y:+.1f}) vxy={vxy:.2f}m/s "
        f"motors=[{motors_str}] {stick_desc} modes=[{arm}|{ang}|{alt}]"
    )


# ============================================================================
#  RESULTS
# ============================================================================

def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E HORIZONTAL FLIGHT TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.tick}")
    print(f"  Max motor value:      {state.max_motor:.3f}")
    print()
    print(f"  Target altitude:      {TARGET_ALTITUDE:.0f}m")
    print(f"  Max altitude:         {state.max_altitude:.1f}m")
    print(f"  Hover target:         {state.hover_target:.1f}m")
    print(f"  Hover max drift:      {state.hover_max_drift:.1f}m (tolerance: {HOVER_TOLERANCE}m)")
    print()
    print(f"  --- Horizontal Flight ---")
    print(f"  FLY_LEFT distance:    {state.fly_left_distance:.1f}m (min: {MIN_HORIZONTAL_DISTANCE}m)")
    print(f"  FLY_RIGHT distance:   {state.fly_right_distance:.1f}m (min: {MIN_HORIZONTAL_DISTANCE}m)")
    print(f"  Alt error in flight:  {state.flight_max_alt_error:.2f}m (tolerance: {FLIGHT_ALT_TOLERANCE}m)")
    print(f"  Final horiz speed:    {state.stabilize_final_vxy:.2f}m/s (threshold: {STABILIZE_VXY_THRESHOLD}m/s)")
    print()
    if state.low_hover_target > 0:
        print(f"  Low hover capture:    {state.low_hover_target:.2f}m")
        print(f"  Low hover max drift:  {state.low_hover_max_drift:.2f}m (tolerance: {LOW_HOVER_DRIFT_TOLERANCE}m)")
    print(f"  Landing altitude:     {state.land_altitude:.2f}m")
    print(f"  Landing velocity:     {state.land_velocity:.2f}m/s")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    # --- Pass/Fail ---
    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if state.max_altitude < TARGET_ALTITUDE * 0.9:
        passed = False
        issues.append(f"Did not reach target altitude ({state.max_altitude:.1f}m < {TARGET_ALTITUDE * 0.9:.1f}m)")

    if state.hover_max_drift > HOVER_TOLERANCE:
        passed = False
        issues.append(f"Hover drift too large ({state.hover_max_drift:.1f}m > {HOVER_TOLERANCE}m)")

    if state.fly_left_distance < MIN_HORIZONTAL_DISTANCE:
        passed = False
        issues.append(f"FLY_LEFT too short ({state.fly_left_distance:.1f}m < {MIN_HORIZONTAL_DISTANCE}m)")

    if state.fly_right_distance < MIN_HORIZONTAL_DISTANCE:
        passed = False
        issues.append(f"FLY_RIGHT too short ({state.fly_right_distance:.1f}m < {MIN_HORIZONTAL_DISTANCE}m)")

    if state.flight_max_alt_error > FLIGHT_ALT_TOLERANCE:
        passed = False
        issues.append(
            f"Altitude error during flight too large "
            f"({state.flight_max_alt_error:.2f}m > {FLIGHT_ALT_TOLERANCE}m)"
        )

    if state.stabilize_final_vxy > STABILIZE_VXY_THRESHOLD:
        issues.append(
            f"Horizontal speed after stabilize too high "
            f"({state.stabilize_final_vxy:.2f}m/s > {STABILIZE_VXY_THRESHOLD}m/s)"
        )

    if state.low_hover_max_drift > LOW_HOVER_DRIFT_TOLERANCE:
        passed = False
        issues.append(
            f"LOW_HOVER drift too large "
            f"({state.low_hover_max_drift:.2f}m > {LOW_HOVER_DRIFT_TOLERANCE}m)"
        )

    if state.land_altitude > 0.20:
        passed = False
        issues.append(f"Did not land properly ({state.land_altitude:.2f}m > 0.20m)")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses from Betaflight")

    if state.max_motor < 0.02:
        passed = False
        issues.append("Motors never spun up")

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
        hsplit name = "Flight Test" {
            viewport name=Viewport pos="drone.world_pos.translate_world(8.0, 8.0, 5.0)" look_at="drone.world_pos" show_grid=#true active=#true
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
    "e2e-flight-test.kdl",
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
            print("  E2E Horizontal Flight Test")
            print(f"  Scenario: takeoff, fly left/right at {TARGET_ALTITUDE}m, near-ground hover, land")
            print(f"  Config: {config.mass}kg quad, ANGLE + ALT_HOLD mode")
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

    # --- Read sensor data ---
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

        # world_pos: [qx, qy, qz, qw, x, y, z]
        # world_vel: [wx, wy, wz, vx, vy, vz]
        s.current_altitude = float(world_pos[6]) if len(world_pos) > 6 else float(world_pos[2])
        s.current_vz = float(world_vel[5]) if len(world_vel) > 5 else float(world_vel[2])
        s.current_x = float(world_pos[4]) if len(world_pos) > 4 else 0.0
        s.current_y = float(world_pos[5]) if len(world_pos) > 5 else 0.0
        s.current_vx = float(world_vel[3]) if len(world_vel) > 3 else 0.0
        s.current_vy = float(world_vel[4]) if len(world_vel) > 4 else 0.0
        s.max_altitude = max(s.max_altitude, s.current_altitude)
        s.baro_altitude = float(baro[0]) if len(baro) > 0 else s.current_altitude
        s.quat_xyzw = world_pos[:4]
        s.gyro_body = gyro
    except RuntimeError as e:
        if tick > 5:
            print(f"[{t:6.1f}s] WARNING: Could not read sensor data: {e}")
        buf.timestamp = t

    # --- Phase transitions ---
    update_phase(s, t, dt=config.sim_time_step)

    # --- Build RC channels ---
    channels = build_rc_channels(s)

    # --- Crash detection ---
    if not s.crash_detected:
        if check_crash(s, t, config.sim_time_step):
            print(f"[{t:6.1f}s] [{s.phase.name:>14}] !! CRASH DETECTED: {s.crash_reason}")
            s.transition(Phase.DISARM, t)

    # --- Status printing ---
    print_status(s, t)

    # --- Build and send packets ---
    fdm = buf.build_fdm()
    fdm.timestamp = t
    rc = RCPacket(timestamp=t, channels=channels)

    try:
        motors = b.step(fdm, rc)
        s.motors = np.array(motors)
        s.max_motor = max(s.max_motor, float(np.max(motors)))
        s.step_count += 1
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        pass

    # --- Done check ---
    if s.phase == Phase.DONE or t >= TEST_TIMEOUT:
        if not s.results_printed:
            elapsed_real = time.time() - _start_time[0]
            print(f"\nSimulation: {t:.1f}s in {elapsed_real:.1f}s ({t/elapsed_real:.1f}x realtime)")
            print_results(s)
        b.stop()
        return


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv

if running_under_editor:
    db_path = "/tmp/e2e_flight_db"
else:
    db_path = "e2e_flight_db"
use_interactive = False

print(f"E2E Horizontal Flight Test")
print(f"  SITL binary: {BETAFLIGHT_PATH.name}")
print(f"  Sim rate: {1.0/config.sim_time_step:.0f}Hz, timeout: {TEST_TIMEOUT}s")
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
