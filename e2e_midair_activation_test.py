#!/usr/bin/env python3
"""
E2E Mid-Air ALTHOLD Activation Test — safety check for low-stick activation.

Validates that activating ALTHOLD mid-air with stick below center does NOT
cause an immediate forced descent. The INITIALIZE state should hold altitude
(targetVelocity=0) when above the near-ground gate (1m), preventing dangerous
altitude loss on activation.

Prerequisites (standard SITL eeprom — same as other E2E tests):
    aux 0 0 0 1700 2100 0 0
    aux 1 1 1 1700 2100 0 0
    aux 2 3 2 1700 2100 0 0
    set failsafe_delay = 200
    set ap_hover_throttle = 1300
    set alt_hold_climb_rate = 200
    set alt_hold_deadband = 50
    save

Assumes default midrc = 1500 and deadband = 50.

Flight phases:
  BOOT(5s) → PRESELECT_ANGLE(ANGLE only, 2s) → ARM(2s) → INITIAL_SETTLE(1500, 1s)
  → CLIMB_ANGLE(1700, wait alt > 5m — climb in ANGLE mode, no ALTHOLD)
  → STABILIZE(stick at 1100, wait |vz| < 0.5 for 0.3s — decel through zero-crossing)
  → ACTIVATE_ALTHOLD(enable ALTHOLD with stick at 1300, 3s — must not dive)
  → HOLD_CHECK(stick at 1500, 5s — verify altitude held vs activation capture)
  → DISARM → DONE

Pass criteria:
  - ACTIVATE_ALTHOLD: no descent spike below -0.3 m/s in first 1s (immediate reaction only)
  - ACTIVATE_ALTHOLD: altitude drop < 0.5m from capture altitude
  - HOLD_CHECK: altitude drift < 1.0m from activation capture altitude over 5s

Run:
    cd elodin && ./run.sh e2e-midair-activation
"""

import math
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
DEADBAND = 50

STICK_CLIMB = MIDRC + 200             # 1700
STICK_HOLD = MIDRC                     # 1500
STICK_LOW_MID = MIDRC - 200           # 1300 — below center
STICK_STABILIZE = 1100                 # low thrust to decelerate and cross vz=0 in ANGLE mode

CLIMB_TARGET_ALT = 5.0
CLIMB_TIMEOUT = 60.0
STABILIZE_VZ_THRESHOLD = 0.5      # m/s — |vz| must be below this before activation
STABILIZE_DWELL = 0.3             # seconds — sustained below threshold (zero-crossing window)
STABILIZE_TIMEOUT = 20.0          # seconds — max time to stabilize
STABILIZE_CEILING = 150.0         # meters — abort if drone goes above this
ACTIVATE_SPIKE_WINDOW = 1.0       # seconds — only check descent spike in first 1s after activation
ACTIVATE_DURATION = 3.0
HOLD_DURATION = 5.0

# Pass/fail thresholds (safety-focused — not precision hold)
ACTIVATE_MAX_DESCENT_VZ = -0.3    # m/s — 100ms avg limit (in first 1s)
ACTIVATE_RAW_DESCENT_GUARD = -0.5 # m/s — single-sample catastrophic guard
ACTIVATE_VZ_WINDOW_S = 0.1       # seconds — 100ms moving average window
ACTIVATE_MAX_ALT_DROP = 1.5       # meters — max altitude loss (safety limit)
HOLD_MAX_DRIFT = 2.5              # meters — max drift during hold (safety limit)

# Soft warning thresholds (track degradation, don't fail)
ACTIVATE_WARN_ALT_DROP = 0.8      # meters — warn if settling sag exceeds this
HOLD_WARN_DRIFT = 1.5             # meters — warn if hold drift exceeds this

TEST_TIMEOUT = 120.0

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
DISARM_DURATION = 1.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT_ANGLE = auto()
    ARM = auto()
    INITIAL_SETTLE = auto()  # post-ARM settle before climb
    CLIMB_ANGLE = auto()
    STABILIZE = auto()       # low stick to decelerate, wait for |vz| < threshold
    ACTIVATE_ALTHOLD = auto()
    HOLD_CHECK = auto()
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

    # Stabilize tracking
    stabilize_dwell_s: float = 0.0     # accumulated time with |vz| < threshold

    # Activation tracking
    activate_capture_alt: float = 0.0
    activate_min_vz_raw: float = float('inf')    # single-sample min (diagnostic + hard guard)
    activate_min_vz_avg: float = float('inf')    # 100ms moving-average min (pass/fail metric)
    activate_vz_window: deque = field(default_factory=lambda: deque())
    activate_max_alt_drop: float = 0.0

    # Hold tracking — uses activate_capture_alt as reference
    hold_max_drift: float = 0.0

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

    if phase == Phase.PRESELECT_ANGLE:
        channels[CH_ANGLE] = MODE_ON
        # ALTHOLD OFF — only ANGLE preselected
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.ARM:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.INITIAL_SETTLE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD

    elif phase == Phase.CLIMB_ANGLE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        # No ALTHOLD — climbing in pure ANGLE mode
        channels[CH_THROTTLE] = STICK_CLIMB

    elif phase == Phase.STABILIZE:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        # Low thrust to decelerate and cross vz=0
        channels[CH_THROTTLE] = STICK_STABILIZE

    elif phase == Phase.ACTIVATE_ALTHOLD:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON  # ALTHOLD activated here!
        channels[CH_THROTTLE] = STICK_LOW_MID  # Stick stays below center

    elif phase == Phase.HOLD_CHECK:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_HOLD

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
            state.transition(Phase.PRESELECT_ANGLE, t)

    elif phase == Phase.PRESELECT_ANGLE:
        if elapsed >= 2.0:
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            state.transition(Phase.INITIAL_SETTLE, t)

    elif phase == Phase.INITIAL_SETTLE:
        if elapsed >= SETTLE_DURATION:
            print(f"[{t:6.1f}s] [    INITIAL_SETTLE] Climbing in ANGLE mode (no ALTHOLD)...")
            state.transition(Phase.CLIMB_ANGLE, t)

    elif phase == Phase.CLIMB_ANGLE:
        if state.current_altitude > CLIMB_TARGET_ALT:
            print(f"[{t:6.1f}s] [       CLIMB_ANGLE] Reached {state.current_altitude:.1f}m. Stabilizing (stick={STICK_STABILIZE})...")
            state.transition(Phase.STABILIZE, t)
        elif elapsed >= CLIMB_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"CLIMB_ANGLE timeout: alt={state.current_altitude:.1f}m < {CLIMB_TARGET_ALT}m"
            state.transition(Phase.DISARM, t)

    elif phase == Phase.STABILIZE:
        # Single phase: low thrust to decelerate, wait for |vz| < threshold during zero-crossing
        if abs(state.current_vz) < STABILIZE_VZ_THRESHOLD:
            state.stabilize_dwell_s += dt
        else:
            state.stabilize_dwell_s = 0.0

        if state.stabilize_dwell_s >= STABILIZE_DWELL:
            state.activate_capture_alt = state.current_altitude
            # Reset activation tracking state
            state.activate_min_vz_raw = float('inf')
            state.activate_min_vz_avg = float('inf')
            state.activate_vz_window.clear()
            print(
                f"[{t:6.1f}s] [         STABILIZE] "
                f"Stabilized (|vz|<{STABILIZE_VZ_THRESHOLD} for {STABILIZE_DWELL}s). "
                f"Activating ALTHOLD with stick at {STICK_LOW_MID}! capture={state.activate_capture_alt:.1f}m"
            )
            state.transition(Phase.ACTIVATE_ALTHOLD, t)
        elif state.current_altitude > STABILIZE_CEILING:
            state.crash_detected = True
            state.crash_reason = f"STABILIZE ceiling: alt={state.current_altitude:.1f}m > {STABILIZE_CEILING}m"
            state.transition(Phase.DISARM, t)
        elif elapsed >= STABILIZE_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"STABILIZE timeout: |vz|={abs(state.current_vz):.2f} after {STABILIZE_TIMEOUT}s"
            state.transition(Phase.DISARM, t)

    elif phase == Phase.ACTIVATE_ALTHOLD:
        # Track descent spike only in first ACTIVATE_SPIKE_WINDOW seconds
        vz_window_ticks = max(1, int(round(ACTIVATE_VZ_WINDOW_S / dt)))
        if elapsed <= ACTIVATE_SPIKE_WINDOW:
            # Raw single-sample min (diagnostic + catastrophic guard)
            state.activate_min_vz_raw = min(state.activate_min_vz_raw, state.current_vz)

            # 100ms moving average
            state.activate_vz_window.append(state.current_vz)
            while len(state.activate_vz_window) > vz_window_ticks:
                state.activate_vz_window.popleft()
            if len(state.activate_vz_window) >= vz_window_ticks:
                avg_vz = sum(state.activate_vz_window) / len(state.activate_vz_window)
                state.activate_min_vz_avg = min(state.activate_min_vz_avg, avg_vz)

        drop = state.activate_capture_alt - state.current_altitude
        state.activate_max_alt_drop = max(state.activate_max_alt_drop, drop)

        if elapsed >= ACTIVATE_DURATION:
            avg_str = f"{state.activate_min_vz_avg:.2f}" if math.isfinite(state.activate_min_vz_avg) else "N/A"
            print(f"[{t:6.1f}s] [  ACTIVATE_ALTHOLD] Done. min_vz_avg={avg_str}m/s raw={state.activate_min_vz_raw:.2f}m/s alt_drop={state.activate_max_alt_drop:.2f}m. Hold check...")
            state.transition(Phase.HOLD_CHECK, t)

    elif phase == Phase.HOLD_CHECK:
        # Use activation capture altitude as hold reference
        drift = abs(state.current_altitude - state.activate_capture_alt)
        state.hold_max_drift = max(state.hold_max_drift, drift)
        if elapsed >= HOLD_DURATION:
            print(f"[{t:6.1f}s] [        HOLD_CHECK] Done. drift={state.hold_max_drift:.2f}m")
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
    thr = channels[CH_THROTTLE]
    althold = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"

    print(
        f"[{t:6.1f}s] [{state.phase.name:>18}] "
        f"alt={state.current_altitude:+7.3f}m vz={state.current_vz:+5.2f}m/s "
        f"T={thr} [{althold}] motors=[{motors_str}]"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E MID-AIR ALTHOLD ACTIVATION — SAFETY CHECK")
    print("  (validates no forced descent, not precision hold)")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    activation_reached = state.activate_capture_alt > 0.0
    print()
    print(f"  --- Activation Phase (stick={STICK_LOW_MID}, below center) ---")
    if activation_reached:
        print(f"  Capture altitude:     {state.activate_capture_alt:.1f}m")
        avg_str = f"{state.activate_min_vz_avg:.2f}" if math.isfinite(state.activate_min_vz_avg) else "N/A"
        raw_str = f"{state.activate_min_vz_raw:.2f}" if math.isfinite(state.activate_min_vz_raw) else "N/A"
        print(f"  Min vz raw (first {ACTIVATE_SPIKE_WINDOW}s):         {raw_str}m/s (hard limit: {ACTIVATE_RAW_DESCENT_GUARD}m/s)")
        print(f"  Min vz 100ms avg (first {ACTIVATE_SPIKE_WINDOW}s):   {avg_str}m/s (limit: {ACTIVATE_MAX_DESCENT_VZ}m/s)")
        print(f"  Max alt drop:         {state.activate_max_alt_drop:.2f}m (limit: {ACTIVATE_MAX_ALT_DROP}m)")
    else:
        print(f"  Activation phase:     N/A (never entered — stabilize failed)")
    print()
    print(f"  --- Hold Check (ref=activation capture) ---")
    if activation_reached:
        print(f"  Reference altitude:   {state.activate_capture_alt:.1f}m")
        print(f"  Hold max drift:       {state.hold_max_drift:.2f}m (limit: {HOLD_MAX_DRIFT}m)")
    else:
        print(f"  Hold check:           N/A")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if not activation_reached:
        passed = False
        issues.append("Activation phase never entered (stabilize failed)")

    if activation_reached and not math.isfinite(state.activate_min_vz_avg):
        passed = False
        issues.append("Activation vz window never filled (test too short or no data)")

    if activation_reached and math.isfinite(state.activate_min_vz_avg) and state.activate_min_vz_avg < ACTIVATE_MAX_DESCENT_VZ:
        passed = False
        issues.append(f"Descent spike (100ms avg): vz={state.activate_min_vz_avg:.2f}m/s < {ACTIVATE_MAX_DESCENT_VZ}m/s")

    if activation_reached and math.isfinite(state.activate_min_vz_raw) and state.activate_min_vz_raw < ACTIVATE_RAW_DESCENT_GUARD:
        passed = False
        issues.append(f"Descent spike (raw, catastrophic): vz={state.activate_min_vz_raw:.2f}m/s < {ACTIVATE_RAW_DESCENT_GUARD}m/s")

    if activation_reached and state.activate_max_alt_drop > ACTIVATE_MAX_ALT_DROP:
        passed = False
        issues.append(f"Altitude loss: {state.activate_max_alt_drop:.2f}m > {ACTIVATE_MAX_ALT_DROP}m")
    elif activation_reached and state.activate_max_alt_drop > ACTIVATE_WARN_ALT_DROP:
        issues.append(f"WARNING: altitude sag {state.activate_max_alt_drop:.2f}m > {ACTIVATE_WARN_ALT_DROP}m (controller settling)")

    if activation_reached and state.hold_max_drift > HOLD_MAX_DRIFT:
        passed = False
        issues.append(f"Hold drift: {state.hold_max_drift:.2f}m > {HOLD_MAX_DRIFT}m")
    elif activation_reached and state.hold_max_drift > HOLD_WARN_DRIFT:
        issues.append(f"WARNING: hold drift {state.hold_max_drift:.2f}m > {HOLD_WARN_DRIFT}m (controller settling)")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses")

    state.test_passed = passed

    print(f"  Status:               {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        print(f"    {'FAIL' if not passed else 'INFO'}: {issue}")
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
        hsplit name = "Mid-Air Activation Test" {
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
    "e2e-midair-activation-test.kdl",
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
            print("  E2E MID-AIR ALTHOLD ACTIVATION — Safety Check")
            print(f"  Validates: no forced descent on mid-air activation (not precision hold)")
            print(f"  Timeout: {TEST_TIMEOUT}s")
            print("=" * 70)
            print()

            bridge_obj.start()
            _bridge[0] = bridge_obj
            print("[  0.0s] [              INIT] Waiting 2s for BF init...")
        except Exception as e:
            print(f"[INIT] ERROR: {e}")
            _bridge[0] = None
            return
        time.sleep(2)

        print("[  0.0s] [              INIT] Warmup...")
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
        print(f"[  0.0s] [              INIT] Warmup: {warmup_ok}/500")
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
        b.stop()
        print_results(s)


# ============================================================================
#  RUN
# ============================================================================

running_under_editor = "--liveness-port" in sys.argv
db_path = "/tmp/e2e_midair_activation_test_db" if running_under_editor else "e2e_midair_activation_test_db"

print(f"E2E MID-AIR ALTHOLD ACTIVATION Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: fly to 5m, activate ALTHOLD with low stick, must not dive")

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
