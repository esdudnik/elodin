#!/usr/bin/env python3
"""
E2E Low-Altitude Horizontal Flight Test — spin-lock regression guard (v10.3.5).

Validates that the ground spin-lock predicate (alt_hold_multirotor.c::
altHoldGroundSpinLockActive) does NOT fire when ALTHOLD is freshly activated
at low altitude after the drone has clearly been airborne in this arm session.

Why this matters: in v10, a pilot toggling ALTHOLD on at low altitude with
horizontal motion could trigger spin-lock → motors clamped → crash. v10.3.1
added the altHoldAirborneSinceArm latch which, once set during the initial
climb, blocks the spin-lock predicate unconditionally for the rest of the
arm session. This test exercises the fresh-activation path with the latch
in place and confirms motors are not clamped.

Scenario (v10.3.5 — ALTHOLD-mediated descent because elodin physics provides
no usable manual-descent regime in ANGLE between hover and free-fall):
  1. CLIMB in ANGLE to >3 m — latch trips (alt>50 cm sustained 200 ms)
  2. Engage ALTHOLD with stick centered (engagement gesture: WAIT_CENTER →
     WAIT_DEPART)
  3. Stick down + ALTHOLD on — ALTHOLD-mediated controlled descent to ~1 m
  4. Brief ALTHOLD-off window so the next ALTHOLD-on is a fresh INITIALIZE
     entry (the original regression scenario), not a continuation. stick=1500
     in ANGLE arrests residual descent velocity; phase exits as soon as
     |vz| recovers OR a hard timeout
  5. LOW_PASS_ACTIVATE: stick=1700 + ALTHOLD on + roll right — fresh
     INITIALIZE at low alt with sustained tilt. Engagement FSM stays in
     WAIT_CENTER (stick=1700, no center-crossing) → drone holds altitude.
     Spin-lock predicate evaluates each tick; airborneSinceArm guard ([6])
     blocks it. Tilt > 5° from roll additionally invalidates the grounded
     predicate ([3]) as defense in depth.
  6. LOW_PASS_HOLD: stick to center — engagement progresses, drone stable.
  7. DISARM → DONE.

Pass criteria:
  PRIMARY: max_motor_during_low_pass > MIN_MAX_MOTOR_DURING_LOW_PASS (0.10)
           — proves spin-lock did NOT clamp motors. This is the regression check.
  SECONDARY: min_alt_during_low_pass > MIN_ALT_DURING_LOW_PASS (0.05 m)
             — drone didn't crash. Distinguishes spin-lock clamp from
             controller-recovery overshoot in failure attribution.
  SECONDARY: alt drop after activation < MAX_ALT_DROP_AFTER_ACTIVATE (0.5 m)
             — bounds reasonable overshoot.

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
STICK_DESCEND = MIDRC - 200         # 1300 — ALTHOLD-mediated descent command
STICK_HOLD = MIDRC
STICK_ROLL_LEFT = MIDRC - 200
STICK_ROLL_RIGHT = MIDRC + 200

CLIMB_TARGET_ALT = 3.0              # m — climb until reaching this; latch trips during climb

# ALTHOLD-mediated descent (v10.3.5) — elodin physics doesn't expose a usable
# manual descent regime in ANGLE: any motor output below hover puts the drone
# into terminal-velocity free-fall (no equilibrium descent). The only way to
# get a controlled low-speed descent is through ALTHOLD itself, which caps
# vertical rate at alt_hold_climb_rate (200 cm/s default).
ALTHOLD_ENGAGE_DURATION = 0.5       # s — engagement gesture phase (stick in deadband ⇒
                                    # WAIT_CENTER→WAIT_DEPART; subsequent stick move triggers ENGAGED)
ALTHOLD_DESCEND_TARGET_ALT = 1.0    # m — transition to OFF_BRIEF when alt drops below this
ALTHOLD_DESCEND_TIMEOUT = 15.0      # s — at ALTHOLD's max -200 cm/s, 3m → 1m is ~1s nominal

# Brief ALTHOLD-off window between descent and fresh activation. Required so
# that LOW_PASS_ACTIVATE genuinely tests a fresh INITIALIZE entry (the original
# regression scenario), not a state already established. stick=1500 at this
# point is aggressive (motors~0.50) — relied on to arrest residual descent vz
# quickly. Phase exits as soon as descent is substantially arrested OR timeout.
ALTHOLD_OFF_ARREST_VZ_THRESHOLD = -0.5   # m/s — vz threshold to declare arrest complete
ALTHOLD_OFF_MIN_DURATION = 0.1            # s — avoid first-tick exit before momentum settles
ALTHOLD_OFF_MAX_DURATION = 0.5            # s — hard timeout safety net

LOW_PASS_DURATION = 3.0             # s — duration of fresh-activation evaluation
LOW_PASS_HOLD_DURATION = 2.0        # s — post-activation stabilization sample window

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
    CLIMB = auto()                 # ANGLE only — drives latch trip during climb past 50cm
    ALTHOLD_ENGAGE_GESTURE = auto() # stick centered + ALTHOLD on — drives WAIT_CENTER→WAIT_DEPART
    ALTHOLD_DESCEND = auto()       # stick low + ALTHOLD on — controlled descent at ~-200 cm/s
    ALTHOLD_OFF_BRIEF = auto()     # ALTHOLD off briefly to allow fresh activation in phase 9
    LOW_PASS_ACTIVATE = auto()     # fresh ALTHOLD activation at low alt with tilt — regression test
    LOW_PASS_HOLD = auto()         # post-activation stabilization sample window
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

    # Low-pass tracking — captured/accumulated during LOW_PASS_ACTIVATE + LOW_PASS_HOLD only
    activate_capture_alt: float = 0.0
    min_alt_during_low_pass: float = float('inf')
    max_motor_during_low_pass: float = 0.0   # if spin-lock clamps motors, this stays low

    # Scenario reach flag — gates spin-lock assertions in print_results.
    # Set ONLY on actual entry to LOW_PASS_ACTIVATE block (not in phase 8 transition)
    # so a flaky phase 8 exit can't produce false-positive coverage in results.
    low_pass_activate_reached: bool = False

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
    elif phase == Phase.ALTHOLD_ENGAGE_GESTURE:
        # ALTHOLD just toggled on with stick in deadband. INITIALIZE starts with
        # engagement = WAIT_CENTER. stick in deadband → WAIT_CENTER → WAIT_DEPART
        # within 1 tick. Drone holds altitude (targetVelocity=0) throughout.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD
    elif phase == Phase.ALTHOLD_DESCEND:
        # Stick away from deadband completes engagement: WAIT_DEPART → ENGAGED →
        # state = IN_PROGRESS. Now stick=STICK_DESCEND in IN_PROGRESS triggers
        # ALTHOLD-mediated descent at the configured climb rate (alt_hold_climb_rate
        # default 200 cm/s). Roll stays centered — drone descends straight down.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_DESCEND
    elif phase == Phase.ALTHOLD_OFF_BRIEF:
        # ALTHOLD off briefly so phase 9 entry is a fresh INITIALIZE rather than
        # a state already established. stick=1500 in ANGLE = motors~0.50 = strong
        # thrust → arrests residual downward velocity from descent in ~0.1s.
        # Roll right starts building tilt for phase 9.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD
        channels[CH_ROLL] = STICK_ROLL_RIGHT
    elif phase == Phase.LOW_PASS_ACTIVATE:
        # THE REGRESSION TEST: fresh ALTHOLD INITIALIZE at low alt with tilt.
        # With v10.3.1 airborne-since-arm latch set during the earlier climb,
        # altHoldGroundSpinLockActive must return false unconditionally — the
        # latch guard ([6]) blocks regardless of grounded predicate.
        # Tilt > 5° from sustained roll-right additionally invalidates
        # isDefinitelyGrounded ([3]) as defense in depth.
        # Engagement FSM stays in WAIT_CENTER (stick=1700, prevSign=0 at FSM
        # reset, no crossing) so drone holds altitude — exactly the state where
        # the predicate is actively evaluated.
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_CLIMB
        channels[CH_ROLL] = STICK_ROLL_RIGHT
    elif phase == Phase.LOW_PASS_HOLD:
        # Pilot brings stick to deadband; engagement progresses normally.
        # Continue tracking min_alt / max_motor for assertion coverage.
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
            print(f"[{t:6.1f}s] [{'CLIMB':>18}] Reached {state.current_altitude:.1f}m. "
                  f"Engaging ALTHOLD for controlled descent...")
            state.transition(Phase.ALTHOLD_ENGAGE_GESTURE, t)
        elif elapsed >= 30.0:
            state.crash_detected = True
            state.crash_reason = f"climb timeout: only reached {state.max_altitude:.2f}m"
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.ALTHOLD_ENGAGE_GESTURE:
        # Stick in deadband for the gesture duration. Engagement FSM:
        #   tick 1: WAIT_CENTER (stick in deadband) → WAIT_DEPART
        #   subsequent ticks: WAIT_DEPART, awaiting stick to leave deadband
        # Drone holds altitude throughout (targetVelocity=0).
        if elapsed >= ALTHOLD_ENGAGE_DURATION:
            print(f"[{t:6.1f}s] [{'ALTHOLD_ENGAGE_GESTURE':>18}] Gesture complete "
                  f"(alt={state.current_altitude:.2f}m). Commanding descent...")
            state.transition(Phase.ALTHOLD_DESCEND, t)

    elif state.phase == Phase.ALTHOLD_DESCEND:
        # Stick=1300 (out of deadband, negative side) completes engagement
        # within 1 tick (WAIT_DEPART → ENGAGED → IN_PROGRESS). Then ALTHOLD
        # commands descent at the configured climb rate.
        if state.current_altitude <= ALTHOLD_DESCEND_TARGET_ALT:
            print(f"[{t:6.1f}s] [{'ALTHOLD_DESCEND':>18}] Reached {state.current_altitude:.2f}m at "
                  f"vz={state.current_vz:+.2f}m/s. Toggling ALTHOLD off briefly...")
            state.transition(Phase.ALTHOLD_OFF_BRIEF, t)
        elif elapsed >= ALTHOLD_DESCEND_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = (
                f"ALTHOLD descend timeout (alt={state.current_altitude:.2f}m "
                f"vz={state.current_vz:+.2f}m/s) — ALTHOLD did not deliver controlled descent"
            )
            state.transition(Phase.DISARM, t)

    elif state.phase == Phase.ALTHOLD_OFF_BRIEF:
        # Wait for descent arrest (vz substantially recovered) OR timeout.
        # Phase 9 entry must happen with low |vz| so the regression test's
        # spin-lock evaluation isn't confounded by controller-recovery overshoot.
        arrested = (state.current_vz > ALTHOLD_OFF_ARREST_VZ_THRESHOLD
                    and elapsed >= ALTHOLD_OFF_MIN_DURATION)
        timed_out = elapsed >= ALTHOLD_OFF_MAX_DURATION
        if arrested or timed_out:
            exit_reason = "arrested" if arrested else "timeout"
            print(f"[{t:6.1f}s] [{'ALTHOLD_OFF_BRIEF':>18}] exit={exit_reason} "
                  f"alt={state.current_altitude:.2f}m vz={state.current_vz:+.2f}m/s. "
                  f"Fresh-activating ALTHOLD at low alt with tilt...")
            state.transition(Phase.LOW_PASS_ACTIVATE, t)

    elif state.phase == Phase.LOW_PASS_ACTIVATE:
        # First-tick latch — set BOTH the reach flag and capture-altitude here
        # (not in the phase 8 exit block) so a flaky phase 8 transition can't
        # falsely advance the gate without phase 9 actually running.
        if not state.low_pass_activate_reached:
            state.low_pass_activate_reached = True
            state.activate_capture_alt = state.current_altitude

        # Critical phase: track motors and altitude
        # If spin-lock incorrectly fires, motors clamp → drone drops fast.
        state.min_alt_during_low_pass = min(state.min_alt_during_low_pass, state.current_altitude)
        if state.motors is not None and len(state.motors) > 0:
            state.max_motor_during_low_pass = max(state.max_motor_during_low_pass, float(np.max(state.motors)))

        if state.current_altitude < MIN_ALT_DURING_LOW_PASS:
            state.crash_detected = True
            # Distinguish spin-lock clamp from controller recovery overshoot.
            # If motors stayed below the running-flight threshold, spin-lock IS
            # firing (motors clamped — the regression we're guarding against).
            # If motors were running normally, the drop is a controller-arrest
            # overshoot from too-aggressive descent velocity at activation —
            # scenario-level issue, not a spin-lock regression.
            if state.max_motor_during_low_pass < MIN_MAX_MOTOR_DURING_LOW_PASS:
                cause = (f"spin-lock clamped motors to max={state.max_motor_during_low_pass:.3f} "
                         f"(below {MIN_MAX_MOTOR_DURING_LOW_PASS}) — REAL spin-lock false-positive")
            else:
                cause = (f"controller recovery overshoot — max_motor={state.max_motor_during_low_pass:.3f} "
                         f"is well above {MIN_MAX_MOTOR_DURING_LOW_PASS}, so spin-lock did NOT fire. "
                         f"Drone arrived at activation with too high downward velocity "
                         f"for ALTHOLD to arrest within available altitude margin")
            state.crash_reason = (
                f"ALT dropped below safety minimum during ALTHOLD activate "
                f"(alt={state.current_altitude:.3f}m, capture={state.activate_capture_alt:.2f}m). "
                f"Cause: {cause}."
            )
            state.transition(Phase.DISARM, t)
        elif elapsed >= LOW_PASS_DURATION:
            print(f"[{t:6.1f}s] [{'LOW_PASS_ACTIVATE':>18}] Done. "
                  f"min_alt={state.min_alt_during_low_pass:.3f}m max_motor={state.max_motor_during_low_pass:.3f}")
            state.transition(Phase.LOW_PASS_HOLD, t)

    elif state.phase == Phase.LOW_PASS_HOLD:
        # Continue tracking — engagement progresses from WAIT_DEPART as stick
        # returns to deadband, but the spin-lock predicate has already been
        # exercised in phase 9. This phase samples post-activation behavior.
        state.min_alt_during_low_pass = min(state.min_alt_during_low_pass, state.current_altitude)
        if state.motors is not None and len(state.motors) > 0:
            state.max_motor_during_low_pass = max(state.max_motor_during_low_pass, float(np.max(state.motors)))
        if elapsed >= LOW_PASS_HOLD_DURATION:
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
    print(f"  Reached LOW_PASS_ACTIVATE: {state.low_pass_activate_reached}")
    if state.low_pass_activate_reached:
        print(f"  Capture altitude:     {state.activate_capture_alt:.2f}m")
        print(f"  Min alt (low-pass):   {state.min_alt_during_low_pass:.3f}m (limit: {MIN_ALT_DURING_LOW_PASS}m)")
        print(f"  Max motor (low-pass): {state.max_motor_during_low_pass:.3f} (must be > {MIN_MAX_MOTOR_DURING_LOW_PASS})")
    print()

    fails = []
    if not state.low_pass_activate_reached:
        # Scenario setup failed — spin-lock evaluation phase never executed.
        # Do NOT run motor/alt-drop assertions: their default values are
        # uninitialized (min_alt=inf, max_motor=0.0) and would produce
        # misleading "spin-lock false-positive" failures.
        fails.append(
            "  FAIL: scenario setup failed — never reached LOW_PASS_ACTIVATE "
            "(spin-lock evaluation phase). Spin-lock assertions skipped."
        )
        if state.crash_reason:
            fails.append(f"    reason: {state.crash_reason}")
    else:
        # Real spin-lock regression assertions — only meaningful if the
        # scenario actually entered the evaluation phase.
        if state.crash_detected:
            fails.append(f"  FAIL: {state.crash_reason}")
        if state.min_alt_during_low_pass < MIN_ALT_DURING_LOW_PASS and not state.crash_detected:
            fails.append(f"  FAIL: drone hit ground during ALTHOLD low-pass (min={state.min_alt_during_low_pass:.3f}m)")
        if state.max_motor_during_low_pass < MIN_MAX_MOTOR_DURING_LOW_PASS:
            # If spin-lock incorrectly fired, all motors would be clamped low;
            # if drone was flying, at least one motor must have been >10%.
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
        hsplit name = "Low-Alt Horizontal Spin-Lock Regression Test" {
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
    "e2e-low-alt-horizontal-test.kdl",
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
            print("  E2E LOW-ALTITUDE HORIZONTAL FLIGHT Test (spin-lock regression)")
            print(f"  Scenario: fly to {CLIMB_TARGET_ALT}m, descend, activate ALTHOLD at low alt with tilt")
            print(f"  Verifies altHoldGroundSpinLockActive does NOT fire during flight")
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
        print(f"[{t:6.1f}s] [{s.phase.name:>18}] alt={s.current_altitude:+6.2f}m "
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
db_path = "/tmp/e2e_low_alt_horizontal_test_db" if running_under_editor else "e2e_low_alt_horizontal_test_db"

print(f"E2E LOW-ALTITUDE HORIZONTAL FLIGHT Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: spin-lock predicate must NOT fire during mid-flight low-pass")

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
