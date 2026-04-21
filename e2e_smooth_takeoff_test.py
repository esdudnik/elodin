#!/usr/bin/env python3
"""
E2E Smooth Takeoff Ramp Test — gradual stick increase from bottom to climb.

Validates that slowly raising throttle from bottom produces a smooth,
predictable takeoff:
  - Drone stays on ground while stick is below center + deadband
  - Drone lifts off smoothly after stick passes center + deadband
  - No sudden motor spikes during ground-to-air transition
  - Integrator windup from -500 works correctly with gradual stick increase

Prerequisites (standard SITL eeprom — same as other E2E tests):
    aux 0 0 0 1700 2100 0 0
    aux 1 1 1 1700 2100 0 0
    aux 2 3 2 1700 2100 0 0
    set failsafe_delay = 200
    set ap_hover_throttle = 1300
    set alt_hold_climb_rate = 200
    set alt_hold_deadband = 50
    save

Assumes default midrc=1500, deadband=50, thr_mid=50, thr_expo=0, LVC off.

Note: The raw-stick to rcCommand mapping in comments below is illustrative
for the default profile only. Assertions use altitude/motor behavior, not
rcCommand values directly.

Flight phases:
  BOOT(5s) → PRESELECT(2s) → ARM(throttle LOW, 2s)
  → RAMP_UP(stick +50 every 3s below center / 2s above center, 1000→1700)
  → HOLD_HIGH(stick 1700, wait alt > 3m, up to 30s)
  → DISARM → DONE

Pass criteria:
  - No premature liftoff: first_liftoff_stick >= MIDRC + DEADBAND (1550)
  - Liftoff detected during RAMP_UP (not only in HOLD_HIGH)
  - Smooth motors: max motor delta over 200ms window < 0.3 while alt < 0.5m
  - Eventual climb: alt > 3m during HOLD_HIGH

Run:
    cd elodin && ./run.sh e2e-smooth-takeoff
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
DEADBAND = 50

RAMP_START = 1000
RAMP_END = 1700
RAMP_STEP = 50
RAMP_DWELL_BELOW = 3.0     # seconds per step while stick < MIDRC
RAMP_DWELL_ABOVE = 2.0     # seconds per step while stick >= MIDRC

GROUND_MAX_ALT = 0.1        # meters — liftoff threshold
LIFTOFF_DWELL = 0.3         # seconds above GROUND_MAX_ALT to confirm liftoff
LIFTOFF_MIN_STICK = MIDRC + DEADBAND  # 1550 — first liftoff must be at or above this
MOTOR_SMOOTH_WINDOW_S = 0.2 # seconds — sliding window for motor delta
MOTOR_SMOOTH_MAX_DELTA = 0.3  # max motor change in window (tunable after first run)
MOTOR_SMOOTH_ALT_LIMIT = 0.5  # only check smoothness below this altitude

CLIMB_TARGET = 3.0
CLIMB_TIMEOUT = 30.0

TEST_TIMEOUT = 100.0

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
ALTHOLD_SETTLE = 2.0
DISARM_DURATION = 1.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    RAMP_UP = auto()
    HOLD_HIGH = auto()
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

    # Ramp state
    current_stick: int = RAMP_START
    stick_change_time: float = 0.0   # when current stick value was set
    ramp_samples: list = field(default_factory=list)  # [(time, stick, alt, max_motor, vz)]

    # Liftoff detection (dwell-based)
    liftoff_above_since: float = 0.0  # time when alt first exceeded threshold (0 = not above)
    first_liftoff_time: float = 0.0
    first_liftoff_stick: int = 0
    first_liftoff_alt: float = 0.0
    liftoff_detected: bool = False

    # Motor smoothness (sliding window)
    motor_history: deque = field(default_factory=lambda: deque())  # (time, max_motor)
    max_motor_delta_ground: float = 0.0  # max delta in window while alt < MOTOR_SMOOTH_ALT_LIMIT

    crash_detected: bool = False
    crash_reason: str = ""

    last_print_time: float = -1.0
    results_printed: bool = False

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

    if phase == Phase.PRESELECT:
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.ARM:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.RAMP_UP:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = state.current_stick

    elif phase == Phase.HOLD_HIGH:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RAMP_END

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
            state.current_stick = RAMP_START
            state.stick_change_time = t
            print(f"[{t:6.1f}s] [           ARM] Armed. Starting ramp from {RAMP_START} to {RAMP_END} (step={RAMP_STEP})...")
            state.transition(Phase.RAMP_UP, t)

    elif phase == Phase.RAMP_UP:
        # Liftoff detection with dwell
        if not state.liftoff_detected:
            if state.current_altitude > GROUND_MAX_ALT:
                if state.liftoff_above_since == 0.0:
                    state.liftoff_above_since = t
                elif (t - state.liftoff_above_since) >= LIFTOFF_DWELL:
                    state.liftoff_detected = True
                    state.first_liftoff_time = state.liftoff_above_since
                    state.first_liftoff_stick = state.current_stick
                    state.first_liftoff_alt = state.current_altitude
                    print(
                        f"[{t:6.1f}s] [       RAMP_UP] "
                        f"LIFTOFF detected! stick={state.current_stick} alt={state.current_altitude:.2f}m"
                    )
            else:
                state.liftoff_above_since = 0.0

        # Motor smoothness tracking (only near ground)
        max_m = float(np.max(state.motors))
        state.motor_history.append((t, max_m))
        window_ticks = int(MOTOR_SMOOTH_WINDOW_S / dt)
        while len(state.motor_history) > window_ticks:
            state.motor_history.popleft()
        if state.current_altitude < MOTOR_SMOOTH_ALT_LIMIT and len(state.motor_history) >= 2:
            motors_in_window = [m for (_, m) in state.motor_history]
            delta = max(motors_in_window) - min(motors_in_window)
            state.max_motor_delta_ground = max(state.max_motor_delta_ground, delta)

        # Dwell time per stick step
        dwell = RAMP_DWELL_BELOW if state.current_stick < MIDRC else RAMP_DWELL_ABOVE
        time_at_stick = t - state.stick_change_time

        if time_at_stick >= dwell:
            # Record sample at end of dwell
            state.ramp_samples.append((
                t, state.current_stick, state.current_altitude,
                float(np.max(state.motors)), state.current_vz
            ))

            if state.current_stick >= RAMP_END:
                # Ramp complete
                print(f"[{t:6.1f}s] [       RAMP_UP] Ramp complete at stick={RAMP_END}. Hold high...")
                state.transition(Phase.HOLD_HIGH, t)
            else:
                # Next step
                state.current_stick = min(state.current_stick + RAMP_STEP, RAMP_END)
                state.stick_change_time = t
                print(
                    f"[{t:6.1f}s] [       RAMP_UP] "
                    f"Stick → {state.current_stick}  alt={state.current_altitude:.2f}m "
                    f"motor={float(np.max(state.motors)):.3f}"
                )

    elif phase == Phase.HOLD_HIGH:
        if state.current_altitude > CLIMB_TARGET:
            print(f"[{t:6.1f}s] [     HOLD_HIGH] Reached {state.current_altitude:.1f}m. Done.")
            state.transition(Phase.DISARM, t)
        elif elapsed >= CLIMB_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"HOLD_HIGH timeout: alt={state.current_altitude:.1f}m < {CLIMB_TARGET}m after {CLIMB_TIMEOUT}s"
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

    liftoff_str = "YES" if state.liftoff_detected else "no"

    print(
        f"[{t:6.1f}s] [{state.phase.name:>14}] "
        f"alt={state.current_altitude:+7.3f}m vz={state.current_vz:+5.2f}m/s "
        f"T={thr} motors=[{motors_str}] liftoff={liftoff_str}"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E SMOOTH TAKEOFF RAMP TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print(f"  MIDRC:                {MIDRC}")
    print(f"  DEADBAND:             {DEADBAND}")
    print()

    # Ramp table
    print(f"  --- Ramp Samples (end of each dwell) ---")
    print(f"  {'Time':>6s}  {'Stick':>5s}  {'Alt(m)':>7s}  {'Motor':>6s}  {'Vz(m/s)':>8s}")
    for (t, stick, alt, motor, vz) in state.ramp_samples:
        marker = " <-- LIFTOFF" if state.liftoff_detected and stick == state.first_liftoff_stick else ""
        print(f"  {t:6.1f}  {stick:5d}  {alt:+7.3f}  {motor:6.3f}  {vz:+8.3f}{marker}")
    print()

    print(f"  --- Liftoff ---")
    if state.liftoff_detected:
        print(f"  First liftoff:        stick={state.first_liftoff_stick} at t={state.first_liftoff_time:.1f}s alt={state.first_liftoff_alt:.2f}m")
        print(f"  Min liftoff stick:    {LIFTOFF_MIN_STICK} (MIDRC+DEADBAND)")
    else:
        print(f"  First liftoff:        NOT DETECTED during RAMP_UP")
    print()

    print(f"  --- Motor Smoothness (alt < {MOTOR_SMOOTH_ALT_LIMIT}m) ---")
    print(f"  Max motor delta (200ms window): {state.max_motor_delta_ground:.3f} (limit: {MOTOR_SMOOTH_MAX_DELTA})")
    print()

    print(f"  --- Climb ---")
    print(f"  Max altitude:         {state.max_altitude:.1f}m (target: {CLIMB_TARGET}m)")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if not state.liftoff_detected:
        passed = False
        issues.append("No liftoff detected during RAMP_UP")

    if state.liftoff_detected and state.first_liftoff_stick < LIFTOFF_MIN_STICK:
        passed = False
        issues.append(
            f"Premature liftoff: stick={state.first_liftoff_stick} < {LIFTOFF_MIN_STICK} "
            f"(MIDRC+DEADBAND). Drone lifted before crossing center+deadband boundary."
        )

    if state.max_motor_delta_ground > MOTOR_SMOOTH_MAX_DELTA:
        passed = False
        issues.append(
            f"Motor spike near ground: delta={state.max_motor_delta_ground:.3f} > {MOTOR_SMOOTH_MAX_DELTA} "
            f"in {MOTOR_SMOOTH_WINDOW_S}s window"
        )

    if state.max_altitude < CLIMB_TARGET:
        passed = False
        issues.append(f"Did not reach climb target: {state.max_altitude:.1f}m < {CLIMB_TARGET}m")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses")

    print(f"  Status:               {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        prefix = "FAIL" if not passed else "INFO"
        print(f"    {prefix}: {issue}")
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
        hsplit name = "Smooth Takeoff Ramp Test" {
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
    "e2e-smooth-takeoff-test.kdl",
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
            print("  E2E SMOOTH TAKEOFF RAMP Test")
            print(f"  Ramp: {RAMP_START} → {RAMP_END} (step={RAMP_STEP})")
            print(f"  Liftoff must be at stick >= {LIFTOFF_MIN_STICK} (MIDRC+DEADBAND)")
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
db_path = "/tmp/e2e_smooth_takeoff_test_db" if running_under_editor else "e2e_smooth_takeoff_test_db"

print(f"E2E SMOOTH TAKEOFF RAMP Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Ramp: {RAMP_START} → {RAMP_END}, liftoff expected at stick >= {LIFTOFF_MIN_STICK}")

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
