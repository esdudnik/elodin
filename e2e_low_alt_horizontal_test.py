#!/usr/bin/env python3
"""
E2E Low-Altitude Horizontal Flight Test — spin-lock regression guard.

Validates that the ground spin-lock predicate (alt_hold_multirotor.c::
altHoldGroundSpinLockActive) does NOT fire during low-altitude horizontal
flight. Without proper safeguards (tilt check, throttle-low check), a drone
flying low and slow could falsely trigger the predicate → motors clamped
mid-flight → crash.

Scenario:
  1. Take off normally to ~3m
  2. Reduce throttle and apply roll to induce low-altitude horizontal motion
  3. While drone is flying low (alt < 0.5m) with horizontal velocity, toggle
     ALTHOLD on
  4. Verify drone does NOT enter spin-lock (would clamp motors)
  5. Pass criterion: drone maintains controllable flight; no catastrophic alt
     drop; motors not clamped during flight

The key test is in the LOW_PASS_ACTIVATE phase: stick is at climb position
(1700) so THROTTLE_LOW=false → spin-lock guard [1] blocks. Tilt from horizontal
motion → spin-lock guard [4] blocks. Predicate must NOT fire.

If spin-lock incorrectly fires, motors get clamped to alt_hold_ground_spin
level, drone loses lift, crashes. Test asserts altitude doesn't suddenly drop.

Prerequisites (standard SITL eeprom):
    aux 0 0 0 1700 2100 0 0
    aux 1 1 1 1700 2100 0 0
    aux 2 3 2 1700 2100 0 0
    set failsafe_delay = 200
    set ap_hover_throttle = 1300
    set alt_hold_climb_rate = 200
    set alt_hold_deadband = 50
    set alt_hold_ground_spin = 50    # opt-in to spin-lock (default 0 = old behavior)
    save

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE on, 2s) → ARM(2s) → SETTLE(1500, 1s)
  → CLIMB(1700, wait alt > 3m)
  → LOW_PASS_DESCEND(stick low, descend toward 0.4m alt)
  → LOW_PASS_ACTIVATE(toggle ALTHOLD with stick HIGH, drone in horizontal motion)
  → LOW_PASS_HOLD(verify drone keeps flying, doesn't spin-lock)
  → DISARM → DONE

Run:
    cd elodin && ./run.sh e2e-low-alt-horizontal
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

MIDRC = 1500
STICK_CLIMB = MIDRC + 200          # 1700
STICK_DESCEND = MIDRC - 200         # 1300
STICK_HOLD = MIDRC
STICK_ROLL_LEFT = MIDRC - 200       # roll command to induce tilt
STICK_ROLL_RIGHT = MIDRC + 200

CLIMB_TARGET_ALT = 3.0              # m — climb until reaching this
LOW_PASS_TARGET_ALT = 0.4           # m — descend until this altitude
LOW_PASS_DURATION = 3.0             # s — duration of low-pass with ALTHOLD on
LOW_PASS_TIMEOUT = 30.0             # s — abort if can't reach low pass

# Pass/fail thresholds
MIN_ALT_DURING_LOW_PASS = 0.05      # m — drone must not bottom out (crash)
MAX_ALT_DROP_AFTER_ACTIVATE = 0.5   # m — drone shouldn't drop > 50cm after ALTHOLD on
MIN_MAX_MOTOR_DURING_LOW_PASS = 0.10  # — at least one motor must be at >10% (proves no spin-lock clamp)

TEST_TIMEOUT = 90.0

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
    CLIMB = auto()
    LOW_PASS_DESCEND = auto()      # stick low, descend toward target alt
    LOW_PASS_ACTIVATE = auto()     # activate ALTHOLD while in horizontal motion at low alt
    LOW_PASS_HOLD = auto()         # verify drone doesn't spin-lock; flying continues
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
    activate_capture_alt: float = 0.0
    min_alt_during_low_pass: float = float('inf')
    max_motor_during_low_pass: float = 0.0   # if spin-lock clamps motors, this stays low

    crash_detected: bool = False
    crash_reason: str = ""

    last_print_time: float = -1.0
    results_printed: bool = False
    test_passed: bool = False

    def transition(self, new_phase: Phase, t: float):
        old = self.phase.name
        self.phase = new_phase
        self.phase_start_time = t
        print(f"[{t:6.1f}s] [{old:>18}] --> [{new_phase.name}]")

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
    elif phase == Phase.LOW_PASS_DESCEND:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = STICK_DESCEND  # 1300, below center → descend
        # apply small roll to induce horizontal motion + tilt
        channels[CH_ROLL] = STICK_ROLL_RIGHT
    elif phase == Phase.LOW_PASS_ACTIVATE:
        # Critical test moment: ALTHOLD on, but stick HIGH (climb), tilt from roll
        # Conditions: THROTTLE_LOW=false, tilt>5° → spin-lock guards [1] and [4] block.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON                  # ALTHOLD just toggled on
        channels[CH_THROTTLE] = STICK_CLIMB              # high stick (NOT THROTTLE_LOW)
        channels[CH_ROLL] = STICK_ROLL_RIGHT             # maintain tilt
    elif phase == Phase.LOW_PASS_HOLD:
        # Pilot brings stick to center to engage ALTHOLD properly
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD
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
            print(f"[{t:6.1f}s] [{'ARM':>18}] Armed. Climbing in ANGLE mode...")
            state.transition(Phase.SETTLE, t)

    elif state.phase == Phase.SETTLE:
        if elapsed >= SETTLE_DURATION:
            state.transition(Phase.CLIMB, t)

    elif state.phase == Phase.CLIMB:
        if state.current_altitude >= CLIMB_TARGET_ALT:
            print(f"[{t:6.1f}s] [{'CLIMB':>18}] Reached {state.current_altitude:.1f}m. Descending for low-pass...")
            state.transition(Phase.LOW_PASS_DESCEND, t)
        elif elapsed >= 30.0:
            state.crash_detected = True
            state.crash_reason = f"climb timeout: only reached {state.max_altitude:.2f}m"
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.LOW_PASS_DESCEND:
        # Descend toward LOW_PASS_TARGET_ALT
        if state.current_altitude <= LOW_PASS_TARGET_ALT and state.current_altitude >= MIN_ALT_DURING_LOW_PASS:
            print(f"[{t:6.1f}s] [{'LOW_PASS_DESCEND':>18}] Reached {state.current_altitude:.2f}m at vz={state.current_vz:.2f}m/s. "
                  f"Activating ALTHOLD with HIGH stick...")
            state.activate_capture_alt = state.current_altitude
            state.transition(Phase.LOW_PASS_ACTIVATE, t)
        elif state.current_altitude < MIN_ALT_DURING_LOW_PASS:
            # Drone hit the ground during descent — test setup issue
            state.crash_detected = True
            state.crash_reason = f"drone bottomed during descent (alt={state.current_altitude:.3f}m)"
            state.transition(Phase.DISARM, t)
        elif elapsed >= LOW_PASS_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"low-pass descend timeout (alt={state.current_altitude:.2f}m)"
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.LOW_PASS_ACTIVATE:
        # Critical phase: track motors and altitude
        # If spin-lock incorrectly fires, motors clamp → drone drops fast.
        state.min_alt_during_low_pass = min(state.min_alt_during_low_pass, state.current_altitude)
        if state.motors is not None and len(state.motors) > 0:
            state.max_motor_during_low_pass = max(state.max_motor_during_low_pass, float(np.max(state.motors)))

        if state.current_altitude < MIN_ALT_DURING_LOW_PASS:
            state.crash_detected = True
            state.crash_reason = (
                f"ALT dropped below safety minimum during ALTHOLD activate "
                f"(alt={state.current_altitude:.3f}m, capture={state.activate_capture_alt:.2f}m). "
                f"Possible spin-lock false-positive."
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= LOW_PASS_DURATION:
            print(f"[{t:6.1f}s] [{'LOW_PASS_ACTIVATE':>18}] Done. "
                  f"min_alt={state.min_alt_during_low_pass:.3f}m max_motor={state.max_motor_during_low_pass:.3f}")
            state.transition(Phase.LOW_PASS_HOLD, t)

    elif state.phase == Phase.LOW_PASS_HOLD:
        # Pilot centers stick to engage ALTHOLD properly
        state.min_alt_during_low_pass = min(state.min_alt_during_low_pass, state.current_altitude)
        if state.motors is not None and len(state.motors) > 0:
            state.max_motor_during_low_pass = max(state.max_motor_during_low_pass, float(np.max(state.motors)))
        if elapsed >= 2.0:
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
    print("  E2E LOW-ALTITUDE HORIZONTAL FLIGHT TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Max altitude:         {state.max_altitude:.2f}m")
    print(f"  Capture altitude:     {state.activate_capture_alt:.2f}m")
    print(f"  Min alt (low-pass):   {state.min_alt_during_low_pass:.3f}m (limit: {MIN_ALT_DURING_LOW_PASS}m)")
    print(f"  Max motor (low-pass): {state.max_motor_during_low_pass:.3f} (must be > {MIN_MAX_MOTOR_DURING_LOW_PASS})")
    print()

    fails = []
    if state.crash_detected:
        fails.append(f"  FAIL: {state.crash_reason}")
    if state.min_alt_during_low_pass < MIN_ALT_DURING_LOW_PASS and not state.crash_detected:
        fails.append(f"  FAIL: drone hit ground during ALTHOLD low-pass (min={state.min_alt_during_low_pass:.3f}m)")
    if state.max_motor_during_low_pass < MIN_MAX_MOTOR_DURING_LOW_PASS:
        # If spin-lock incorrectly fired, all motors would be clamped low; if drone was flying,
        # at least one motor must have been >10%.
        fails.append(
            f"  FAIL: max_motor during low-pass = {state.max_motor_during_low_pass:.3f} < {MIN_MAX_MOTOR_DURING_LOW_PASS}. "
            f"Possible spin-lock false-positive (motors clamped during flight)."
        )
    drop = max(0.0, state.activate_capture_alt - state.min_alt_during_low_pass)
    if drop > MAX_ALT_DROP_AFTER_ACTIVATE:
        fails.append(f"  FAIL: alt dropped {drop:.3f}m after activation (limit: {MAX_ALT_DROP_AFTER_ACTIVATE}m)")

    if fails:
        print("  Status:               FAIL")
        for f in fails:
            print(f)
    else:
        print("  Status:               PASS")
        state.test_passed = True

    print("=" * 70)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    config = DEFAULT_CONFIG
    config.set_as_global()

    world = el.World()
    drone_id = world.spawn(Drone())
    world.recipe(el.s10.PyRecipe.process(
        name="Betaflight SITL",
        cmd=str(BETAFLIGHT_PATH),
        cwd=str(BETAFLIGHT_DIR),
    ))

    sensor_system = create_sensor_system(config)
    physics_system = create_physics_system(config)

    state = TestState()
    bridge = BetaflightSyncBridge(config)

    def post_step(ctx, world_state):
        nonlocal state
        state.tick += 1
        state.sim_time = state.tick * config.dt
        state.step_count += 1

        body = world_state.get_body(drone_id)
        state.current_altitude = float(body.pos[2])
        state.current_vz = float(body.vel[2])
        state.max_altitude = max(state.max_altitude, state.current_altitude)
        if state.motors is None or len(state.motors) == 0:
            state.motors = np.zeros(4)
        try:
            motor_values = np.array(world_state.motor_command, dtype=np.float64)
            state.motors = motor_values
        except Exception:
            pass

        rc_channels = build_rc_channels(state)
        bridge.send_rc(RCPacket(channels=rc_channels))
        bridge.exchange_with_betaflight(world_state, state.sim_time)
        update_phase(state, state.sim_time, config.dt)

        if state.sim_time - state.last_print_time >= 1.0 or state.last_print_time < 0:
            print(f"[{state.sim_time:6.1f}s] [{state.phase.name:>18}] alt={state.current_altitude:+6.2f}m "
                  f"vz={state.current_vz:+5.2f}m/s motors=[{','.join(f'{m:.3f}' for m in state.motors)}]")
            state.last_print_time = state.sim_time

        if state.phase == Phase.DONE:
            print_results(state)
            return False
        if state.sim_time >= TEST_TIMEOUT:
            state.crash_reason = "test timeout"
            state.crash_detected = True
            print_results(state)
            return False
        return True

    world.set_post_step(post_step)
    world.run(systems=[sensor_system, physics_system])

    if state.test_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
