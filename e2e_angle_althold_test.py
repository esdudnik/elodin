#!/usr/bin/env python3
"""
E2E ANGLE+ALTHOLD Test: Preselect modes, arm, takeoff, climb, hover, descend, land.

Tests Betaflight's ANGLE + ALT_HOLD with iNav-style preselection before arming:
- ANGLE and ALTHOLD switches are set BEFORE arming (like iNav multirotor)
- BF registers switch states but does not activate ALT_HOLD_MODE until armed
- On first armed tick: ALT_HOLD activates, takeoff prep triggers
- Betaflight SITL runs the full flight controller
- Elodin runs the physics simulation
- Communication via UDP lockstep at 1kHz

Flight phases:
  BOOT(5s) -> PRESELECT(ANGLE+ALTHOLD on, ARM off, 2s)
  -> ARM(2s) -> SETTLE(1s) -> CLIMB(to 5.5m)
  -> TOP_APPROACH(slow climb to 6.8m) -> HOVER(10s) -> DESCEND(to 2m)
  -> RAMP_APPROACH(throttle ramp 1380->1450, to 0.5m + 1s dwell)
  -> LOW_HOVER(10s near ground) -> LAND(contact-based) -> DISARM -> DONE

Landing uses contact-based disarm: altitude < 0.1m AND |vz| < 0.2m/s held for 0.5s.

The test script only sets RC stick positions and AUX channel switches.
Betaflight controls throttle internally via its altitude PID controller.

Run:
    cd elodin && ./run.sh e2e-angle-althold          # headless (recommended)
    cd elodin && ./run.sh e2e-angle-althold-editor   # with 3D viewport

Prerequisites:
    1. Build Betaflight SITL: cd betaflight && make TARGET=SITL DEBUG=GDB
    2. Configure eeprom (one-time):
        aux 0 0 0 1700 2100 0 0    # AUX1 (rcData[4]) -> ARM
        aux 1 1 1 1700 2100 0 0    # AUX2 (rcData[5]) -> ANGLE
        aux 2 3 2 1700 2100 0 0    # AUX3 (rcData[6]) -> ALTHOLD
        set failsafe_delay = 200
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
# This copy lives inside elodin/, imports from examples/betaflight-sitl/
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

TARGET_ALTITUDE = 7.0        # meters — climb target
HOVER_DURATION = 10.0        # seconds — hold at target altitude
HOVER_TOLERANCE = 0.5        # meters — max acceptable drift during hover
LAND_MAX_VELOCITY = 1.5      # m/s — max acceptable landing speed
TEST_TIMEOUT = 200.0         # seconds — total simulation time limit

# Top approach — slow climb before hover to let BF estimator converge
TOP_APPROACH_ALTITUDE = 5.5  # meters — switch from fast climb to slow climb (early entry for convergence)
TOP_APPROACH_THROTTLE = 1570 # just above deadband — slow climb (~10cm/s), keeps isAdjustingAltitude=true
TOP_APPROACH_VZ = 0.2        # m/s — max |vz| to enter HOVER
TOP_APPROACH_MIN_ALT = 6.8   # meters — minimum altitude to enter HOVER (one-sided gate)
TOP_APPROACH_DWELL = 0.5     # seconds — vz must stay below threshold for this long before HOVER entry
TOP_APPROACH_TIMEOUT = 30.0  # seconds — max time in top approach (timeout = FAIL)
HOVER_TARGET_TOLERANCE = 2.0 # meters — hover target must be within this of TARGET_ALTITUDE to pass

# Approach & landing (estimator-lag-aware thresholds)
APPROACH_ALTITUDE = 2.0      # meters — switch from coarse to gentle descent
APPROACH_THROTTLE = 1380     # just below deadband — gentle ~0.25m/s descent
APPROACH_TIMEOUT = 60.0      # seconds — max time in approach phase

# Ramp approach — altitude-based throttle ramp (iNav-style descent velocity ramping)
# Throttle ramps linearly from APPROACH_THROTTLE at APPROACH_ALTITUDE to RAMP_THROTTLE_HIGH near ground
RAMP_THROTTLE_HIGH = 1450    # throttle at low altitude (still below deadband 1450-1550)
RAMP_APPROACH_TIMEOUT = 60.0 # seconds — max time in ramp approach

# LOW_HOVER — near-ground hold
LOW_HOVER_ENTRY_ALT = 0.5   # meters — max altitude to enter LOW_HOVER
LOW_HOVER_ENTRY_VZ = 0.2    # m/s — max |vz| to enter LOW_HOVER
LOW_HOVER_DWELL = 1.0       # seconds — conditions must hold before entry (estimator convergence)
LOW_HOVER_DURATION = 10.0   # seconds — hold time near ground
LOW_HOVER_TIMEOUT = 30.0    # seconds — max time waiting to enter LOW_HOVER
LOW_HOVER_DRIFT_TOLERANCE = 0.5  # meters — max drift from capture altitude during LOW_HOVER

# BF auto-disarm detection (Phase 3: BF owns landing)
# Detect BF disarm by motors going to zero while ARM is still commanded.
BF_DISARM_MOTOR_THRESHOLD = 0.01  # all motors must be below this
BF_DISARM_DWELL = 0.3             # seconds — motors must stay zero for this long

# RC channel indices (AETR + AUX)
CH_ROLL = 0
CH_PITCH = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_ARM = 4       # AUX1 (rcData[4]) — mapped to BOXARM (permanentId=0) via 'aux 0 0 0 1700 2100'
CH_ANGLE = 5     # AUX2 (rcData[5]) — mapped to BOXANGLE (permanentId=1) via 'aux 1 1 1 1700 2100'
CH_ALTHOLD = 6   # AUX3 (rcData[6]) — mapped to BOXALTHOLD (permanentId=3) via 'aux 2 3 2 1700 2100'

# RC values
RC_CENTER = 1500
RC_LOW = 1000
MODE_ON = 1800   # within 1700-2100 activation range
MODE_OFF = 1000  # outside activation range

# ALT_HOLD throttle zones (from BF alt_hold_multirotor.c):
#   < 1400 (40% stick) = descend
#   1400-1600 (40-60%) = hold altitude (deadband)
#   > 1600 (60% stick) = climb
ALTHOLD_CLIMB = 1700
ALTHOLD_HOLD = 1500
ALTHOLD_DESCEND = 1300

# Phase durations
BOOT_DURATION = 5.0     # BF gyro calibration
ARM_DURATION = 2.0      # let arming settle
ALTHOLD_SETTLE = 2.0    # let ALT_HOLD mode engage + hover ramp build up + hover ramp build up
LAND_TIMEOUT = 15.0     # max time for controlled landing descent
DISARM_DURATION = 1.0
CLIMB_TIMEOUT = 90.0    # max time to reach target altitude
DESCEND_TIMEOUT = 60.0  # max time to descend

# Crash / runaway detection
CRASH_MOTOR_ASYMMETRY = 0.7    # max - min motor difference threshold
CRASH_GROUND_STUCK_TIME = 3.0  # seconds on ground with asymmetric motors = crash
CRASH_MAX_VELOCITY = 8.0       # m/s — vertical speed indicating out of control
ALTITUDE_CEILING = 50.0        # meters — absolute altitude limit, fail immediately if exceeded
RUNAWAY_CLIMB_TIME = 2.0       # seconds — sustained positive vz during descent phases = runaway


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()       # ANGLE+ALTHOLD switches on, ARM off — switch preselection only
    ARM = auto()             # ARM on — ALTHOLD activates on first armed tick, takeoff prep triggers
    SETTLE = auto()          # throttle CENTER — enables stick adjustment
    CLIMB = auto()
    TOP_APPROACH = auto()    # slow climb (throttle 1630) — lets estimator converge before hover
    HOVER = auto()
    DESCEND = auto()         # coarse descent at ~1m/s (throttle 1300)
    APPROACH = auto()        # gentle descent at ~0.25m/s (throttle 1380)
    RAMP_APPROACH = auto()   # altitude-based throttle ramp (1380→1450) — iNav-style descent velocity ramping
    LOW_HOVER = auto()       # near-ground hold (~0.3-0.5m) — tests ground effect zone
    LAND = auto()            # gentle descent to touchdown + contact-based disarm
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
    top_approach_dwell: float = 0.0  # how long vz has been below threshold in TOP_APPROACH
    low_hover_dwell: float = 0.0     # how long conditions met for LOW_HOVER entry
    low_hover_target: float = 0.0    # altitude at LOW_HOVER capture
    low_hover_max_drift: float = 0.0 # max drift from capture during LOW_HOVER

    # Landing tracking
    land_altitude: float = 0.0
    land_velocity: float = 0.0
    ground_contact_time: float = 0.0  # how long on_ground conditions have been met
    bf_disarm_dwell: float = 0.0      # how long BF motors have been zero while ARM is still commanded

    # Crash detection
    crash_detected: bool = False
    crash_reason: str = ""
    ground_stuck_time: float = 0.0  # how long on ground with asymmetric motors
    runaway_climb_time: float = 0.0  # how long vz has been positive during descent phases

    # Diagnostics
    last_print_time: float = -1.0
    results_printed: bool = False

    # Attitude diagnostics (Elodin physics truth)
    quat_xyzw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    gyro_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body-frame gyro rad/s (from drone.gyro)

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>14}] --> [{new_phase.name}]")

    def phase_elapsed(self, t: float) -> float:
        return t - self.phase_start_time


def build_rc_channels(state: TestState) -> np.ndarray:
    """Build RC channel array based on current test phase."""
    channels = np.full(MAX_RC_CHANNELS, RC_CENTER, dtype=np.uint16)
    channels[CH_THROTTLE] = RC_LOW
    channels[CH_ARM] = MODE_OFF
    channels[CH_ANGLE] = MODE_OFF
    channels[CH_ALTHOLD] = MODE_OFF

    phase = state.phase

    if phase == Phase.BOOT:
        pass  # everything off

    elif phase == Phase.PRESELECT:
        # Switch preselection: ANGLE+ALTHOLD on, ARM off.
        # BF sees switch states but ALT_HOLD_MODE does not activate (gated by ARMED).
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.ARM:
        # ARM on — ALTHOLD activates on first armed tick.
        # Low throttle + near ground → takeoff prep triggers here.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.SETTLE:
        # Center throttle to enable BF's allowStickAdjustment flag.
        # BF requires stick to pass through center before accepting climb/descend.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD  # 1500 = center

    elif phase == Phase.CLIMB:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_CLIMB  # above 60% = climb

    elif phase == Phase.TOP_APPROACH:
        # Slow climb above deadband — lets BF estimator converge before hover capture
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = TOP_APPROACH_THROTTLE  # above deadband, slow climb

    elif phase == Phase.HOVER:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD  # center = hold

    elif phase == Phase.DESCEND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_DESCEND  # below 40% = coarse descent

    elif phase == Phase.APPROACH:
        # Gentle descent to let BF estimator converge before landing
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = APPROACH_THROTTLE  # just below deadband = gentle descent

    elif phase == Phase.RAMP_APPROACH:
        # Altitude-based throttle ramp: slower descent as altitude decreases (iNav-style)
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        # Linear ramp from APPROACH_THROTTLE at APPROACH_ALTITUDE to RAMP_THROTTLE_HIGH near ground
        alt = max(state.current_altitude, 0.0)
        progress = (APPROACH_ALTITUDE - alt) / (APPROACH_ALTITUDE - LOW_HOVER_ENTRY_ALT)
        progress = max(0.0, min(1.0, progress))
        throttle = int(APPROACH_THROTTLE + progress * (RAMP_THROTTLE_HIGH - APPROACH_THROTTLE))
        channels[CH_THROTTLE] = throttle

    elif phase == Phase.LOW_HOVER:
        # Near-ground hold — stick centered to capture altitude
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = ALTHOLD_HOLD  # center = hold altitude

    elif phase == Phase.LAND:
        # Controlled landing: keep ALT_HOLD active with gentle descend.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = APPROACH_THROTTLE  # gentle descent to touchdown

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.DONE:
        channels[CH_THROTTLE] = RC_LOW

    return channels


def check_crash(state: TestState, t: float, dt: float) -> bool:
    """Detect crash conditions. Returns True if crash detected."""
    if state.phase in (Phase.BOOT, Phase.PRESELECT, Phase.ARM, Phase.SETTLE, Phase.DISARM, Phase.DONE):
        return False

    motors = state.motors
    if np.max(motors) < 0.02:
        return False  # motors not spinning yet

    motor_asymmetry = float(np.max(motors) - np.min(motors))
    on_ground = state.current_altitude < 0.5

    # Crash: on ground with highly asymmetric motors (drone flipped)
    if on_ground and motor_asymmetry > CRASH_MOTOR_ASYMMETRY:
        state.ground_stuck_time += dt
        if state.ground_stuck_time >= CRASH_GROUND_STUCK_TIME:
            state.crash_detected = True
            state.crash_reason = (
                f"Drone flipped on ground. "
                f"Motors asymmetry: {motor_asymmetry:.2f} "
                f"(motors=[{motors[0]:.3f},{motors[1]:.3f},{motors[2]:.3f},{motors[3]:.3f}]). "
                f"Stuck for {state.ground_stuck_time:.1f}s"
            )
            return True
    else:
        state.ground_stuck_time = 0.0

    # Crash: extreme vertical velocity (out of control)
    if abs(state.current_vz) > CRASH_MAX_VELOCITY:
        state.crash_detected = True
        state.crash_reason = (
            f"Extreme vertical velocity: vz={state.current_vz:+.1f}m/s "
            f"(limit: {CRASH_MAX_VELOCITY}m/s). Flight out of control."
        )
        return True

    # Runaway: altitude ceiling exceeded
    if state.current_altitude > ALTITUDE_CEILING:
        state.crash_detected = True
        state.crash_reason = (
            f"Altitude ceiling exceeded: alt={state.current_altitude:.1f}m "
            f"(limit: {ALTITUDE_CEILING}m). Runaway climb."
        )
        return True

    # Runaway: sustained climb during descent phases
    descent_phases = (Phase.DESCEND, Phase.APPROACH, Phase.RAMP_APPROACH, Phase.LOW_HOVER, Phase.LAND)
    if state.phase in descent_phases:
        if state.current_vz > 0.5:  # climbing at > 0.5 m/s during descent
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


def update_phase(state: TestState, t: float, dt: float = 0.001):
    """Phase transition logic based on current state and flight data."""
    phase = state.phase
    elapsed = state.phase_elapsed(t)

    # Check for crash in all active flight phases
    if check_crash(state, t, 0.001):
        print(
            f"[{t:6.1f}s] [{phase.name:>14}] "
            f"!! CRASH DETECTED: {state.crash_reason}"
        )
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
        # Wait 1s with stick at center for allowStickAdjustment to latch
        if elapsed >= 1.0:
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        if state.current_altitude >= TOP_APPROACH_ALTITUDE:
            print(
                f"[{t:6.1f}s] [         CLIMB] "
                f"Near target: alt={state.current_altitude:.1f}m — switching to TOP_APPROACH"
            )
            state.transition(Phase.TOP_APPROACH, t)
        elif elapsed >= CLIMB_TIMEOUT:
            print(
                f"[{t:6.1f}s] [         CLIMB] "
                f"TIMEOUT after {CLIMB_TIMEOUT:.0f}s: "
                f"alt={state.current_altitude:.1f}m (target={TARGET_ALTITUDE}m)"
            )
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
                f"At target & slow for {state.top_approach_dwell:.1f}s: "
                f"alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s — switching to HOVER"
            )
            state.hover_target = state.current_altitude
            state.transition(Phase.HOVER, t)
        elif elapsed >= TOP_APPROACH_TIMEOUT:
            print(
                f"[{t:6.1f}s] [  TOP_APPROACH] "
                f"TIMEOUT after {TOP_APPROACH_TIMEOUT:.0f}s: "
                f"alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s"
            )
            # Timeout is a failure — don't transition to HOVER at wrong altitude
            state.crash_detected = True
            state.crash_reason = (
                f"TOP_APPROACH timeout: alt={state.current_altitude:.1f}m "
                f"(target={TARGET_ALTITUDE}m) vz={state.current_vz:.2f}m/s"
            )
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOVER:
        drift = abs(state.current_altitude - state.hover_target)
        state.hover_altitudes.append(state.current_altitude)
        state.hover_max_drift = max(state.hover_max_drift, drift)

        if elapsed >= HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [         HOVER] "
                f"Complete ({HOVER_DURATION:.0f}s). Max drift: {state.hover_max_drift:.1f}m"
            )
            state.transition(Phase.DESCEND, t)

    elif phase == Phase.DESCEND:
        if state.current_altitude <= APPROACH_ALTITUDE:
            print(
                f"[{t:6.1f}s] [       DESCEND] "
                f"Reached {APPROACH_ALTITUDE:.0f}m — switching to RAMP_APPROACH"
            )
            state.transition(Phase.RAMP_APPROACH, t)
        elif elapsed >= DESCEND_TIMEOUT:
            print(
                f"[{t:6.1f}s] [       DESCEND] "
                f"TIMEOUT after {DESCEND_TIMEOUT:.0f}s: alt={state.current_altitude:.1f}m"
            )
            state.transition(Phase.RAMP_APPROACH, t)

    elif phase == Phase.APPROACH:
        # APPROACH decelerates from ~1m/s to ~0.25m/s, then hands off to RAMP_APPROACH
        if state.current_altitude <= APPROACH_ALTITUDE - 0.5:  # 1.5m
            print(
                f"[{t:6.1f}s] [      APPROACH] "
                f"Starting ramp: alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s — switching to RAMP_APPROACH"
            )
            state.transition(Phase.RAMP_APPROACH, t)
        elif elapsed >= APPROACH_TIMEOUT:
            print(
                f"[{t:6.1f}s] [      APPROACH] "
                f"TIMEOUT after {APPROACH_TIMEOUT:.0f}s: alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s"
            )
            state.transition(Phase.RAMP_APPROACH, t)

    elif phase == Phase.RAMP_APPROACH:
        # Altitude-based throttle ramp — descent slows as altitude decreases
        vz_abs = abs(state.current_vz)
        if state.current_altitude <= LOW_HOVER_ENTRY_ALT and vz_abs < LOW_HOVER_ENTRY_VZ:
            state.low_hover_dwell += dt
        else:
            state.low_hover_dwell = 0.0

        if state.low_hover_dwell >= LOW_HOVER_DWELL:
            print(
                f"[{t:6.1f}s] [ RAMP_APPROACH] "
                f"Low & slow for {state.low_hover_dwell:.1f}s: "
                f"alt={state.current_altitude:.2f}m vz={state.current_vz:.2f}m/s — switching to LOW_HOVER"
            )
            state.transition(Phase.LOW_HOVER, t)
        elif elapsed >= RAMP_APPROACH_TIMEOUT:
            print(
                f"[{t:6.1f}s] [ RAMP_APPROACH] "
                f"TIMEOUT after {RAMP_APPROACH_TIMEOUT:.0f}s: alt={state.current_altitude:.1f}m vz={state.current_vz:.2f}m/s"
            )
            state.transition(Phase.LAND, t)

    elif phase == Phase.LOW_HOVER:
        # Track drift from capture altitude
        if state.low_hover_target == 0.0:
            state.low_hover_target = state.current_altitude  # capture on first tick
        drift = abs(state.current_altitude - state.low_hover_target)
        state.low_hover_max_drift = max(state.low_hover_max_drift, drift)

        if elapsed >= LOW_HOVER_DURATION:
            print(
                f"[{t:6.1f}s] [     LOW_HOVER] "
                f"Complete ({LOW_HOVER_DURATION:.0f}s). alt={state.current_altitude:.2f}m "
                f"capture={state.low_hover_target:.2f}m drift={state.low_hover_max_drift:.2f}m"
            )
            state.transition(Phase.LAND, t)

    elif phase == Phase.LAND:
        # BF auto-disarm detection: ARM stays commanded, wait for BF to drop motors.
        # Motors going to zero while ARM is still on = BF landing detector triggered disarm.
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


def print_status(state: TestState, t: float):
    """Print periodic status update (every 1 second)."""
    if t - state.last_print_time < 1.0:
        return
    state.last_print_time = t

    phase_name = state.phase.name
    motors_str = ",".join(f"{m:.3f}" for m in state.motors)

    # Describe current stick position
    channels = build_rc_channels(state)
    throttle = channels[CH_THROTTLE]
    althold_on = channels[CH_ALTHOLD] == MODE_ON
    if throttle == RC_LOW:
        stick_desc = f"T={throttle} (IDLE)"
    elif althold_on and throttle > 1600:
        stick_desc = f"T={throttle} (CLIMB)"
    elif althold_on and throttle < 1400:
        stick_desc = f"T={throttle} (DESCEND)"
    elif althold_on:
        stick_desc = f"T={throttle} (HOLD)"
    else:
        stick_desc = f"T={throttle}"

    # Mode flags
    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    ang = "ANG" if channels[CH_ANGLE] == MODE_ON else "---"
    alt = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"

    # Baro: altitude reading sent to BF (with noise), and derived pressure
    baro_pressure = 101325.0 - 12.0 * state.baro_altitude
    baro_str = f"baro={state.baro_altitude:+7.2f}m ({baro_pressure:.0f}Pa)"

    # Elodin attitude: quaternion -> roll/pitch in degrees (ENU/FLU frame)
    qx, qy, qz, qw = state.quat_xyzw[0], state.quat_xyzw[1], state.quat_xyzw[2], state.quat_xyzw[3]
    elo_roll_deg = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    elo_pitch_deg = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)))
    # drone.gyro: body-frame gyro in rad/s (same sensor path that feeds the FDM packet to BF)
    gx, gy, gz = state.gyro_body[0], state.gyro_body[1], state.gyro_body[2]

    print(
        f"[{t:6.1f}s] [{phase_name:>14}] "
        f"alt={state.current_altitude:+7.2f}m "
        f"vz={state.current_vz:+6.2f}m/s "
        f"{baro_str} "
        f"motors=[{motors_str}] "
        f"{stick_desc} "
        f"modes=[{arm}|{ang}|{alt}]"
    )
    # Attitude + body-frame gyro diagnostic (same sensor path as FDM packet to BF)
    # BF gyroADCf is in deg/s FRD. Elodin gyro is in rad/s FLU.
    # Conversion: BF_gyroR ≈ Elodin_gx*RAD2DEG, BF_gyroP ≈ -Elodin_gy*RAD2DEG (pitch inverted)
    print(
        f"[ELODIN_ATT] roll={elo_roll_deg:+6.1f}° pitch={elo_pitch_deg:+6.1f}° "
        f"gyro=[{np.degrees(gx):+7.1f}, {np.degrees(gy):+7.1f}, {np.degrees(gz):+7.1f}]°/s"
    )


def print_results(state: TestState):
    """Print final test results."""
    print()
    print("=" * 70)
    print("  E2E ALT_HOLD TEST RESULTS")
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
    if state.low_hover_target > 0:
        print(f"  Low hover capture:    {state.low_hover_target:.2f}m")
        print(f"  Low hover max drift:  {state.low_hover_max_drift:.2f}m (tolerance: {LOW_HOVER_DRIFT_TOLERANCE}m)")
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

    if state.max_altitude < TARGET_ALTITUDE * 0.9:
        passed = False
        issues.append(
            f"Did not reach target altitude "
            f"({state.max_altitude:.1f}m < {TARGET_ALTITUDE * 0.9:.1f}m)"
        )

    if abs(state.hover_target - TARGET_ALTITUDE) > HOVER_TARGET_TOLERANCE:
        passed = False
        issues.append(
            f"Hover target too far from requested altitude "
            f"(hover_target={state.hover_target:.1f}m, requested={TARGET_ALTITUDE}m, "
            f"tolerance={HOVER_TARGET_TOLERANCE}m)"
        )

    if state.hover_max_drift > HOVER_TOLERANCE:
        passed = False
        issues.append(
            f"Hover drift too large "
            f"({state.hover_max_drift:.1f}m > {HOVER_TOLERANCE}m)"
        )

    if state.low_hover_max_drift > LOW_HOVER_DRIFT_TOLERANCE:
        passed = False
        issues.append(
            f"LOW_HOVER drift too large "
            f"({state.low_hover_max_drift:.2f}m > {LOW_HOVER_DRIFT_TOLERANCE}m, "
            f"capture={state.low_hover_target:.2f}m)"
        )

    if state.land_altitude > 0.20:
        passed = False
        issues.append(
            f"Did not land properly "
            f"({state.land_altitude:.2f}m > 0.20m)"
        )

    if state.land_velocity > LAND_MAX_VELOCITY:
        issues.append(
            f"Landing velocity high "
            f"({state.land_velocity:.2f}m/s > {LAND_MAX_VELOCITY}m/s)"
        )

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

# Kill stale BF processes from previous runs
if "--no-s10" not in sys.argv:
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "betaflight_SITL"], capture_output=True, timeout=5)
        time.sleep(0.1)
    except Exception:
        pass

# Create world
world = el.World()

# Drone entity
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

# Ground entity (static, camera orbit target for editor)
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

# Editor schematic (used when running with 'elodin editor')
world.schematic(
    """
    tabs {
        hsplit name = "E2E Test" {
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
    "e2e-test.kdl",
)

# Physics + sensor systems
physics = create_physics_system(config)
sensors = create_sensor_system(config)
system = physics | sensors

# Register Betaflight SITL process (s10 manages its lifecycle)
betaflight_recipe = el.s10.PyRecipe.process(
    name="Betaflight SITL",
    cmd=str(BETAFLIGHT_PATH),
    cwd=str(BETAFLIGHT_DIR),
)
world.recipe(betaflight_recipe)


# ============================================================================
#  POST-STEP CALLBACK (called every physics tick at 1kHz)
# ============================================================================

_bridge = [None]
_sensor_buf = [None]
_state = [None]
_start_time = [None]
max_ticks = int(TEST_TIMEOUT / config.sim_time_step)


def e2e_post_step(tick: int, ctx: el.StepContext):
    """Post-step callback implementing lockstep sync and test phase logic."""

    # --- Lazy initialization (first tick) ---
    if _bridge[0] is None:
        try:
            bridge_obj = BetaflightSyncBridge(timeout_ms=100)
            _sensor_buf[0] = SensorDataBuffer()
            _state[0] = TestState()
            _start_time[0] = time.time()

            print()
            print("=" * 70)
            print("  E2E ANGLE+ALTHOLD Test (preselect before arm)")
            print(f"  Scenario: preselect modes, arm, climb to {TARGET_ALTITUDE}m, "
                  f"hover {HOVER_DURATION}s, descend, land")
            print(f"  Config: {config.mass}kg quad, ANGLE + ALT_HOLD preselected")
            print(f"  Controller: iNav-style cascaded (sqrt + velocity PID)")
            print(f"  Timeout: {TEST_TIMEOUT}s")
            print("=" * 70)
            print()

            bridge_obj.start()
            _bridge[0] = bridge_obj  # only set after successful start
            print("[  0.0s] [          INIT] Waiting 2s for Betaflight initialization...")
        except Exception as e:
            print(f"[INIT] ERROR: Failed to initialize bridge: {e}")
            _bridge[0] = None  # ensure retry on next tick
            return
        time.sleep(2)

        # Warmup: send 500ms of idle packets to prime BF's RC and gyro
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

    # After test completes, skip all processing
    if s.results_printed:
        return

    s.tick = tick
    s.sim_time = tick * config.sim_time_step
    t = s.sim_time

    # --- Read sensor data from physics ---
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
            world_pos=world_pos,
            world_vel=world_vel,
            accel=accel,
            gyro=gyro,
            baro=baro,
            timestamp=t,
        )

        # world_pos layout: [qx, qy, qz, qw, x, y, z]
        # world_vel layout: [wx, wy, wz, vx, vy, vz]
        s.current_altitude = float(world_pos[6]) if len(world_pos) > 6 else float(world_pos[2])
        s.current_vz = float(world_vel[5]) if len(world_vel) > 5 else float(world_vel[2])
        s.max_altitude = max(s.max_altitude, s.current_altitude)
        # baro layout: [altitude_m] — barometer reading with noise
        s.baro_altitude = float(baro[0]) if len(baro) > 0 else s.current_altitude
        # Attitude diagnostics: quaternion and body-frame gyro (same sensor path sent to BF)
        s.quat_xyzw = world_pos[:4]  # [qx, qy, qz, qw]
        s.gyro_body = gyro           # [gx, gy, gz] body-frame gyro rad/s (from drone.gyro)
    except RuntimeError as e:
        if tick > 5:
            print(f"[{t:6.1f}s] WARNING: Could not read sensor data: {e}")
        buf.timestamp = t

    # --- Phase transitions ---
    update_phase(s, t, dt=config.sim_time_step)

    # --- Build RC channels ---
    channels = build_rc_channels(s)

    # --- Lockstep: send FDM+RC, wait for motors ---
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

    # --- Status output ---
    print_status(s, t)

    # --- Check completion ---
    if s.phase == Phase.DONE and not s.results_printed:
        b.stop()
        elapsed = time.time() - _start_time[0]
        print(f"\nSimulation: {s.sim_time:.1f}s in {elapsed:.1f}s "
              f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)")
        print_results(s)
        s.results_printed = True

    # --- Timeout ---
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

# Detect if running under s10/editor (s10 adds --liveness-port when spawning sim)
running_under_editor = "--liveness-port" in sys.argv

# When running under the editor:
# - Use /tmp/ for db_path to avoid triggering s10's file watcher
#   (writes to elodin root directory cause watch-mode restart loop)
# - Keep interactive=False so sim runs at full speed (not throttled to editor fps)
#   After script exits, s10 watch loop just waits for file events (no restart
#   since db is in /tmp/ and no other files change in the elodin directory)
if running_under_editor:
    db_path = "/tmp/e2e_test_db"
else:
    db_path = "e2e_test_db"
use_interactive = False

print(f"E2E ANGLE+ALTHOLD Test (preselect before arm)")
print(f"  SITL binary: {BETAFLIGHT_PATH.name}")
print(f"  Sim rate: {1.0/config.sim_time_step:.0f}Hz, timeout: {TEST_TIMEOUT}s")
print(f"  Scenario: takeoff -> climb to {TARGET_ALTITUDE}m -> hover {HOVER_DURATION}s -> descend -> land")
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
