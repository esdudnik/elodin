#!/usr/bin/env python3
"""
E2E POSHOLD Test: Takeoff, hover, acquire heading, enable POSHOLD, test hold + stick override.

Tests Betaflight's existing POSHOLD implementation with ANGLE + ALT_HOLD:
- Normal takeoff and climb to hover altitude with ANGLE + ALTHOLD
- Acquire GPS heading via forward flight (compass not driven in SITL)
- Enable POSHOLD — drone should hold XY position
- Verify hold radius after settle
- Stick override: fly left with POSHOLD still on
- Verify braking and re-capture at new position
- Standard descent + BF auto-disarm landing

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE+ALTHOLD, POSHOLD off, 2s) → ARM(2s) → SETTLE(1s)
  → CLIMB(to 5.5m) → TOP_APPROACH(to 6.8m) → HOVER(3s)
  → ACQUIRE_HEADING(pitch forward 3s, settle 2s — build GPS heading confidence)
  → ENABLE_POSHOLD → POSHOLD_SETTLE(3s)
  → HOLD_1(10s — verify XY drift bounded)
  → MOVE_LEFT(roll=1400, 3s — pilot override)
  → MOVE_SETTLE(3s — braking)
  → HOLD_2(10s — verify re-capture)
  → DESCEND → LAND(BF auto-disarm) → DISARM → DONE

Run:
    cd elodin && ./run.sh e2e-poshold          # headless
    cd elodin && ./run.sh e2e-poshold-editor   # with 3D viewport

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
        set pos_hold_without_mag = ON
        aux 0 0 0 1700 2100 0 0
        aux 1 1 1 1700 2100 0 0
        aux 2 3 2 1700 2100 0 0
        aux 3 27 3 1700 2100 0 0
        aux 4 11 4 1700 2100 0 0
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
HOVER_DURATION = 3.0         # short hover before heading acquisition
HOVER_TOLERANCE = 0.5
TEST_TIMEOUT = 200.0

# Top approach
TOP_APPROACH_ALTITUDE = 5.5
TOP_APPROACH_THROTTLE = 1570
TOP_APPROACH_VZ = 0.2
TOP_APPROACH_MIN_ALT = 6.8
TOP_APPROACH_DWELL = 0.5
TOP_APPROACH_TIMEOUT = 30.0

# Heading acquisition — fly forward to build GPS heading confidence
HEADING_FLY_DURATION = 3.0      # seconds of forward pitch
HEADING_FLY_PITCH = 1400        # gentle forward pitch (below center = forward)
HEADING_SETTLE_VXY = 0.3        # m/s — must decay below this before proceeding
HEADING_SETTLE_DWELL = 1.0      # seconds — vxy must stay below threshold
HEADING_SETTLE_TIMEOUT = 10.0   # seconds — max wait for settle

# POSHOLD phases
POSHOLD_SETTLE_VXY = 0.3        # m/s — must decay below this before capturing hold point
POSHOLD_SETTLE_DWELL = 1.0      # seconds — vxy must stay below threshold
POSHOLD_SETTLE_TIMEOUT = 10.0   # seconds — max wait for settle
HOLD_DURATION = 10.0            # seconds to measure hold quality
HOLD_RADIUS_TOLERANCE = 2.0     # meters — max XY drift from capture point
HOLD_SPEED_TOLERANCE = 0.5      # m/s — max horizontal speed during hold
MOVE_DURATION = 5.0             # seconds of stick override
MOVE_ROLL = 1300                # stronger left bank (POSHOLD fights back, need more authority)
MOVE_SETTLE_VXY = 0.3           # m/s — must decay below this after move
MOVE_SETTLE_DWELL = 1.0         # seconds — vxy must stay below threshold
MOVE_SETTLE_TIMEOUT = 10.0      # seconds — max wait for braking
MIN_MOVE_DISTANCE = 1.0         # meters — minimum distance moved during override
POSHOLD_ALT_TOLERANCE = 0.8     # meters — altitude error during XY test

# Descent & landing
DESCEND_TIMEOUT = 60.0
LAND_TIMEOUT = 15.0
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
CH_FAILSAFE = 7
CH_POSHOLD = 8      # AUX5 — BOXPOSHOLD (permanentId=11)

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
ALTITUDE_CEILING = 50.0
CRASH_MAX_VELOCITY = 8.0


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
    ACQUIRE_HEADING = auto()     # fly forward to build GPS heading confidence
    HEADING_SETTLE = auto()      # settle after forward flight
    ENABLE_POSHOLD = auto()      # turn POSHOLD on
    POSHOLD_SETTLE = auto()      # let POSHOLD stabilize
    HOLD_1 = auto()              # measure hold quality
    MOVE_LEFT = auto()           # stick override with POSHOLD on
    MOVE_SETTLE = auto()         # braking after stick release
    HOLD_2 = auto()              # measure re-capture quality
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

    # POSHOLD tracking
    hold_1_capture_x: float = 0.0
    hold_1_capture_y: float = 0.0
    hold_1_max_radius: float = 0.0
    hold_1_max_speed: float = 0.0
    move_start_x: float = 0.0
    move_start_y: float = 0.0
    move_distance: float = 0.0
    hold_2_capture_x: float = 0.0
    hold_2_capture_y: float = 0.0
    hold_2_max_radius: float = 0.0
    hold_2_max_speed: float = 0.0
    poshold_max_alt_error: float = 0.0
    poshold_hover_alt: float = 0.0
    heading_settle_dwell: float = 0.0   # behavior-based settle counter
    poshold_settle_dwell: float = 0.0   # behavior-based settle counter
    move_settle_dwell: float = 0.0      # behavior-based settle counter

    # Landing tracking
    land_altitude: float = 0.0
    land_velocity: float = 0.0
    bf_disarm_dwell: float = 0.0

    # Crash detection
    crash_detected: bool = False
    crash_reason: str = ""

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

    def distance_from(self, x: float, y: float) -> float:
        return ((self.current_x - x)**2 + (self.current_y - y)**2) ** 0.5


# ============================================================================
#  RC CHANNEL BUILDER
# ============================================================================

def build_rc_channels(state: TestState) -> np.ndarray:
    channels = np.full(MAX_RC_CHANNELS, RC_CENTER, dtype=np.uint16)
    channels[CH_THROTTLE] = RC_LOW
    channels[CH_ARM] = MODE_OFF
    channels[CH_ANGLE] = MODE_OFF
    channels[CH_ALTHOLD] = MODE_OFF
    channels[CH_POSHOLD] = MODE_OFF

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

    elif phase == Phase.ACQUIRE_HEADING:
        # Fly forward to build GPS heading confidence. POSHOLD still off.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_PITCH] = HEADING_FLY_PITCH  # forward pitch

    elif phase == Phase.HEADING_SETTLE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase in (Phase.ENABLE_POSHOLD, Phase.POSHOLD_SETTLE, Phase.HOLD_1, Phase.HOLD_2):
        # POSHOLD on, sticks centered
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_POSHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD

    elif phase == Phase.MOVE_LEFT:
        # Stick override with POSHOLD still on
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_POSHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD
        channels[CH_ROLL] = MOVE_ROLL

    elif phase == Phase.MOVE_SETTLE:
        # Sticks centered, POSHOLD on — braking
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_POSHOLD] = MODE_ON
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
#  CRASH DETECTION
# ============================================================================

def check_crash(state: TestState, t: float, dt: float) -> bool:
    if state.phase in (Phase.BOOT, Phase.PRESELECT, Phase.ARM, Phase.SETTLE, Phase.DISARM, Phase.DONE):
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


# ============================================================================
#  PHASE TRANSITIONS
# ============================================================================

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
            print(f"[{t:6.1f}s] [     PRESELECT] Switches set (ANGLE+ALTHOLD). Arming...")
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(f"[{t:6.1f}s] [           ARM] Armed. ALTHOLD active. alt={state.current_altitude:.1f}m")
            state.transition(Phase.SETTLE, t)

    elif phase == Phase.SETTLE:
        if elapsed >= 1.0:
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        if state.current_altitude >= TOP_APPROACH_ALTITUDE:
            state.transition(Phase.TOP_APPROACH, t)
        elif elapsed >= CLIMB_TIMEOUT:
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
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)

    elif phase == Phase.HOVER:
        drift = abs(state.current_altitude - state.hover_target)
        state.hover_max_drift = max(state.hover_max_drift, drift)
        state.hover_altitudes.append(state.current_altitude)
        if elapsed >= HOVER_DURATION:
            print(f"[{t:6.1f}s] [         HOVER] Stable. Acquiring GPS heading...")
            state.transition(Phase.ACQUIRE_HEADING, t)

    elif phase == Phase.ACQUIRE_HEADING:
        # Fly forward to build GPS heading confidence (canUseGPSHeading)
        if elapsed >= HEADING_FLY_DURATION:
            vxy = state.horizontal_speed()
            print(f"[{t:6.1f}s] [ACQUIRE_HEADING] Forward flight done. vxy={vxy:.2f}m/s. Settling...")
            state.transition(Phase.HEADING_SETTLE, t)

    elif phase == Phase.HEADING_SETTLE:
        vxy = state.horizontal_speed()
        if vxy < HEADING_SETTLE_VXY:
            state.heading_settle_dwell += dt
        else:
            state.heading_settle_dwell = 0.0
        if state.heading_settle_dwell >= HEADING_SETTLE_DWELL:
            print(f"[{t:6.1f}s] [HEADING_SETTLE] Settled (vxy={vxy:.2f}m/s). Enabling POSHOLD...")
            state.transition(Phase.ENABLE_POSHOLD, t)
        elif elapsed >= HEADING_SETTLE_TIMEOUT:
            print(f"[{t:6.1f}s] [HEADING_SETTLE] SETTLE TIMEOUT (vxy={vxy:.2f}m/s still above threshold). Enabling POSHOLD anyway...")
            state.transition(Phase.ENABLE_POSHOLD, t)

    elif phase == Phase.ENABLE_POSHOLD:
        # Immediate transition — POSHOLD switch set in RC channels
        state.poshold_hover_alt = state.current_altitude
        print(
            f"[{t:6.1f}s] [ENABLE_POSHOLD] "
            f"POSHOLD ON at ({state.current_x:.1f},{state.current_y:.1f}) alt={state.current_altitude:.1f}m"
        )
        state.transition(Phase.POSHOLD_SETTLE, t)

    elif phase == Phase.POSHOLD_SETTLE:
        alt_err = abs(state.current_altitude - state.poshold_hover_alt)
        state.poshold_max_alt_error = max(state.poshold_max_alt_error, alt_err)
        vxy = state.horizontal_speed()
        if vxy < POSHOLD_SETTLE_VXY:
            state.poshold_settle_dwell += dt
        else:
            state.poshold_settle_dwell = 0.0
        if state.poshold_settle_dwell >= POSHOLD_SETTLE_DWELL:
            state.hold_1_capture_x = state.current_x
            state.hold_1_capture_y = state.current_y
            print(
                f"[{t:6.1f}s] [POSHOLD_SETTLE] "
                f"Settled (vxy={vxy:.2f}m/s). "
                f"Capture=({state.hold_1_capture_x:.1f},{state.hold_1_capture_y:.1f}). "
                f"Measuring hold..."
            )
            state.transition(Phase.HOLD_1, t)
        elif elapsed >= POSHOLD_SETTLE_TIMEOUT:
            state.hold_1_capture_x = state.current_x
            state.hold_1_capture_y = state.current_y
            print(
                f"[{t:6.1f}s] [POSHOLD_SETTLE] "
                f"SETTLE TIMEOUT (vxy={vxy:.2f}m/s still above threshold). "
                f"Capture=({state.hold_1_capture_x:.1f},{state.hold_1_capture_y:.1f}). May be inaccurate."
            )
            state.transition(Phase.HOLD_1, t)

    elif phase == Phase.HOLD_1:
        alt_err = abs(state.current_altitude - state.poshold_hover_alt)
        state.poshold_max_alt_error = max(state.poshold_max_alt_error, alt_err)
        radius = state.distance_from(state.hold_1_capture_x, state.hold_1_capture_y)
        state.hold_1_max_radius = max(state.hold_1_max_radius, radius)
        vxy = state.horizontal_speed()
        state.hold_1_max_speed = max(state.hold_1_max_speed, vxy)
        if elapsed >= HOLD_DURATION:
            print(
                f"[{t:6.1f}s] [        HOLD_1] "
                f"Complete. max_radius={state.hold_1_max_radius:.2f}m "
                f"max_speed={state.hold_1_max_speed:.2f}m/s. Moving left..."
            )
            state.move_start_x = state.current_x
            state.move_start_y = state.current_y
            state.transition(Phase.MOVE_LEFT, t)

    elif phase == Phase.MOVE_LEFT:
        alt_err = abs(state.current_altitude - state.poshold_hover_alt)
        state.poshold_max_alt_error = max(state.poshold_max_alt_error, alt_err)
        dist = state.distance_from(state.move_start_x, state.move_start_y)
        state.move_distance = max(state.move_distance, dist)
        if elapsed >= MOVE_DURATION:
            dx = state.current_x - state.move_start_x
            dy = state.current_y - state.move_start_y
            print(
                f"[{t:6.1f}s] [     MOVE_LEFT] "
                f"Complete. distance={state.move_distance:.1f}m "
                f"displacement=({dx:+.1f},{dy:+.1f}) "
                f"pos=({state.current_x:.1f},{state.current_y:.1f}). Settling..."
            )
            state.transition(Phase.MOVE_SETTLE, t)

    elif phase == Phase.MOVE_SETTLE:
        alt_err = abs(state.current_altitude - state.poshold_hover_alt)
        state.poshold_max_alt_error = max(state.poshold_max_alt_error, alt_err)
        vxy = state.horizontal_speed()
        if vxy < MOVE_SETTLE_VXY:
            state.move_settle_dwell += dt
        else:
            state.move_settle_dwell = 0.0
        if state.move_settle_dwell >= MOVE_SETTLE_DWELL:
            state.hold_2_capture_x = state.current_x
            state.hold_2_capture_y = state.current_y
            print(
                f"[{t:6.1f}s] [  MOVE_SETTLE] "
                f"Braked (vxy={vxy:.2f}m/s). "
                f"Capture=({state.hold_2_capture_x:.1f},{state.hold_2_capture_y:.1f}). "
                f"Measuring hold..."
            )
            state.transition(Phase.HOLD_2, t)
        elif elapsed >= MOVE_SETTLE_TIMEOUT:
            state.hold_2_capture_x = state.current_x
            state.hold_2_capture_y = state.current_y
            print(
                f"[{t:6.1f}s] [  MOVE_SETTLE] "
                f"SETTLE TIMEOUT (vxy={vxy:.2f}m/s still above threshold). "
                f"Capture=({state.hold_2_capture_x:.1f},{state.hold_2_capture_y:.1f}). May be inaccurate."
            )
            state.transition(Phase.HOLD_2, t)

    elif phase == Phase.HOLD_2:
        alt_err = abs(state.current_altitude - state.poshold_hover_alt)
        state.poshold_max_alt_error = max(state.poshold_max_alt_error, alt_err)
        radius = state.distance_from(state.hold_2_capture_x, state.hold_2_capture_y)
        state.hold_2_max_radius = max(state.hold_2_max_radius, radius)
        vxy = state.horizontal_speed()
        state.hold_2_max_speed = max(state.hold_2_max_speed, vxy)
        if elapsed >= HOLD_DURATION:
            print(
                f"[{t:6.1f}s] [        HOLD_2] "
                f"Complete. max_radius={state.hold_2_max_radius:.2f}m "
                f"max_speed={state.hold_2_max_speed:.2f}m/s. Descending..."
            )
            state.transition(Phase.DESCEND, t)

    elif phase == Phase.DESCEND:
        if state.current_altitude < 0.5:
            state.transition(Phase.LAND, t)
        elif elapsed >= DESCEND_TIMEOUT:
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
                f"BF auto-disarm detected: alt={state.current_altitude:.2f}m "
                f"motors=0 for {state.bf_disarm_dwell:.1f}s"
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= LAND_TIMEOUT:
            state.land_altitude = state.current_altitude
            state.land_velocity = abs(state.current_vz)
            print(
                f"[{t:6.1f}s] [          LAND] "
                f"FAIL: BF auto-disarm not observed after {LAND_TIMEOUT:.0f}s"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.DISARM:
        if elapsed >= DISARM_DURATION:
            state.transition(Phase.DONE, t)


# ============================================================================
#  STATUS PRINTING
# ============================================================================

def print_status(state: TestState, t: float):
    if int(t) <= state.last_print_time:
        return
    state.last_print_time = int(t)

    phase_name = state.phase.name
    motors_str = ",".join(f"{m:.3f}" for m in state.motors)

    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]
    poshold_on = channels[CH_POSHOLD] == MODE_ON

    if throttle <= 1050:
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
    pos = " PH" if poshold_on else ""

    vxy = state.horizontal_speed()
    roll = channels[CH_ROLL]
    pitch = channels[CH_PITCH]
    rp_str = f" R={roll} P={pitch}" if (roll != RC_CENTER or pitch != RC_CENTER) else ""

    print(
        f"[{t:6.1f}s] [{phase_name:>14}] "
        f"alt={state.current_altitude:+7.2f}m vz={state.current_vz:+5.2f}m/s "
        f"xy=({state.current_x:+.1f},{state.current_y:+.1f}) vxy={vxy:.2f}m/s "
        f"motors=[{motors_str}] {stick_desc}{rp_str} modes=[{arm}|{ang}|{alt}]{pos}"
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
    print("  E2E POSHOLD TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print(f"  Max motor value:      {state.max_motor:.3f}")
    print()
    print(f"  Target altitude:      {TARGET_ALTITUDE:.0f}m")
    print(f"  Max altitude:         {state.max_altitude:.1f}m")
    print(f"  Hover target:         {state.hover_target:.1f}m")
    print(f"  Hover max drift:      {state.hover_max_drift:.1f}m (tolerance: {HOVER_TOLERANCE}m)")
    print()
    print(f"  --- POSHOLD ---")
    print(f"  HOLD_1 max radius:    {state.hold_1_max_radius:.2f}m (tolerance: {HOLD_RADIUS_TOLERANCE}m)")
    print(f"  HOLD_1 max speed:     {state.hold_1_max_speed:.2f}m/s (tolerance: {HOLD_SPEED_TOLERANCE}m/s)")
    print(f"  Move distance:        {state.move_distance:.1f}m (min: {MIN_MOVE_DISTANCE}m)")
    print(f"  HOLD_2 max radius:    {state.hold_2_max_radius:.2f}m (tolerance: {HOLD_RADIUS_TOLERANCE}m)")
    print(f"  HOLD_2 max speed:     {state.hold_2_max_speed:.2f}m/s (tolerance: {HOLD_SPEED_TOLERANCE}m/s)")
    print(f"  Alt error in POSHOLD: {state.poshold_max_alt_error:.2f}m (tolerance: {POSHOLD_ALT_TOLERANCE}m)")
    print()
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

    if state.hold_1_max_radius > HOLD_RADIUS_TOLERANCE:
        passed = False
        issues.append(f"HOLD_1 drift too large ({state.hold_1_max_radius:.2f}m > {HOLD_RADIUS_TOLERANCE}m)")

    if state.hold_1_max_speed > HOLD_SPEED_TOLERANCE:
        passed = False
        issues.append(f"HOLD_1 speed too high ({state.hold_1_max_speed:.2f}m/s > {HOLD_SPEED_TOLERANCE}m/s)")

    if state.hold_2_max_radius > HOLD_RADIUS_TOLERANCE:
        passed = False
        issues.append(f"HOLD_2 drift too large ({state.hold_2_max_radius:.2f}m > {HOLD_RADIUS_TOLERANCE}m)")

    if state.hold_2_max_speed > HOLD_SPEED_TOLERANCE:
        passed = False
        issues.append(f"HOLD_2 speed too high ({state.hold_2_max_speed:.2f}m/s > {HOLD_SPEED_TOLERANCE}m/s)")

    if state.move_distance < MIN_MOVE_DISTANCE:
        passed = False
        issues.append(f"Move too short ({state.move_distance:.1f}m < {MIN_MOVE_DISTANCE}m)")

    if state.poshold_max_alt_error > POSHOLD_ALT_TOLERANCE:
        passed = False
        issues.append(f"Alt error too large ({state.poshold_max_alt_error:.2f}m > {POSHOLD_ALT_TOLERANCE}m)")

    if state.land_altitude > 0.20:
        passed = False
        issues.append(f"Did not land properly ({state.land_altitude:.2f}m > 0.20m)")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses from Betaflight")

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
        hsplit name = "POSHOLD Test" {
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
    "e2e-poshold-test.kdl",
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
            print("  E2E POSHOLD Test")
            print(f"  Scenario: climb to {TARGET_ALTITUDE}m, acquire heading, "
                  f"enable POSHOLD, test hold + override")
            print(f"  Config: {config.mass}kg quad, ANGLE + ALT_HOLD + POSHOLD")
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

        # world_pos: [qx, qy, qz, qw, x, y, z]
        # world_vel: [wx, wy, wz, vx, vy, vz]
        s.current_altitude = float(world_pos[6]) if len(world_pos) > 6 else float(world_pos[2])
        s.current_vz = float(world_vel[5]) if len(world_vel) > 5 else float(world_vel[2])
        s.current_x = float(world_pos[4]) if len(world_pos) > 6 else 0.0
        s.current_y = float(world_pos[5]) if len(world_pos) > 6 else 0.0
        s.current_vx = float(world_vel[3]) if len(world_vel) > 5 else 0.0
        s.current_vy = float(world_vel[4]) if len(world_vel) > 5 else 0.0
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

    # Timeout
    if tick >= max_ticks - 1 and not s.results_printed:
        print(f"\n[{t:6.1f}s] TEST TIMEOUT ({TEST_TIMEOUT}s) in phase {s.phase.name}")
        s.land_altitude = s.current_altitude
        s.land_velocity = abs(s.current_vz)
        b.stop()
        print_results(s)


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv

if running_under_editor:
    db_path = "/tmp/e2e_poshold_test_db"
else:
    db_path = "e2e_poshold_test_db"
use_interactive = False

print(f"E2E POSHOLD Test")
print(f"  SITL binary: {BETAFLIGHT_PATH.name}")
print(f"  Sim rate: {1.0/config.sim_time_step:.0f}Hz, timeout: {TEST_TIMEOUT}s")
print(f"  Scenario: takeoff -> hover -> acquire heading -> POSHOLD hold/move/hold -> land")
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
