#!/usr/bin/env python3
"""
E2E Ground Idle Test — regression for INITIALIZE thrust buildup bug.

Validates that ARMING with ALTHOLD + low throttle does NOT cause gradual
thrust buildup leading to unintended liftoff. This is the "moon takeoff" bug
where the ALTHOLD controller in INITIALIZE state would accumulate upward
thrust at idle throttle, eventually causing the drone to lift off without
any pilot input.

Flight phases:
  BOOT(5s) → PRESELECT(ANGLE+ALTHOLD, 2s) → ARM(throttle LOW, 2s)
  → GROUND_IDLE(15s — pilot does nothing, throttle stays LOW)
  → DISARM → DONE

Pass criteria:
  - z < 0.1m throughout GROUND_IDLE (drone never lifts off)
  - Motors do not show sustained upward trend during GROUND_IDLE
    (max in last 3s must not exceed max in first 3s + 0.02)

Run:
    cd elodin && ./run.sh e2e-ground-idle
"""

import os
import sys
import time
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

GROUND_IDLE_DURATION = 15.0     # seconds — how long to sit at low throttle
GROUND_IDLE_MAX_ALT = 0.1       # meters — must stay below this
TREND_WINDOW = 3.0              # seconds — window size for motor trend check
TREND_IGNORE_INITIAL = 1.0      # seconds — skip first N seconds of GROUND_IDLE (transients)
TREND_MAX_INCREASE = 0.02       # max allowed motor increase from early to late window
MOTOR_SOFT_WARNING = 0.15       # soft warning threshold (doesn't fail)

TEST_TIMEOUT = 60.0

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
    GROUND_IDLE = auto()    # sit at low throttle, do nothing
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
    max_altitude_during_idle: float = 0.0

    # Motor trend tracking during GROUND_IDLE
    motor_samples: list = field(default_factory=list)  # list of (time, max_motor) tuples

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

    elif phase == Phase.GROUND_IDLE:
        # Pilot does nothing — armed, ALTHOLD on, throttle LOW
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = RC_LOW

    elif phase == Phase.DISARM:
        channels[CH_THROTTLE] = RC_LOW

    return channels


# ============================================================================
#  PHASE TRANSITIONS
# ============================================================================

def update_phase(state: TestState, t: float, dt: float):
    phase = state.phase
    elapsed = state.phase_elapsed(t)

    if phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.transition(Phase.PRESELECT, t)

    elif phase == Phase.PRESELECT:
        if elapsed >= ALTHOLD_SETTLE:
            print(f"[{t:6.1f}s] [     PRESELECT] Switches set. Arming...")
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(
                f"[{t:6.1f}s] [           ARM] "
                f"Armed. Entering GROUND_IDLE (pilot does nothing for {GROUND_IDLE_DURATION}s)..."
            )
            state.transition(Phase.GROUND_IDLE, t)

    elif phase == Phase.GROUND_IDLE:
        # Track max altitude during idle
        state.max_altitude_during_idle = max(state.max_altitude_during_idle, state.current_altitude)

        # Record motor samples for trend analysis
        max_m = float(np.max(state.motors))
        state.motor_samples.append((elapsed, max_m))

        # Hard fail: drone lifted off
        if state.current_altitude > GROUND_IDLE_MAX_ALT:
            state.crash_detected = True
            state.crash_reason = (
                f"Drone lifted off during GROUND_IDLE "
                f"(alt={state.current_altitude:.2f}m > {GROUND_IDLE_MAX_ALT}m "
                f"at t={elapsed:.1f}s)"
            )
            print(f"[{t:6.1f}s] [   GROUND_IDLE] FAIL: {state.crash_reason}")
            state.transition(Phase.DISARM, t)
            return

        if elapsed >= GROUND_IDLE_DURATION:
            print(
                f"[{t:6.1f}s] [   GROUND_IDLE] "
                f"Complete. max_alt={state.max_altitude_during_idle:.3f}m "
                f"max_motor={state.max_motor:.3f}"
            )
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

    arm = "ARM" if channels[CH_ARM] == MODE_ON else "---"
    ang = "ANG" if channels[CH_ANGLE] == MODE_ON else "---"
    alt = "ALT" if channels[CH_ALTHOLD] == MODE_ON else "---"

    print(
        f"[{t:6.1f}s] [{state.phase.name:>14}] "
        f"alt={state.current_altitude:+7.3f}m vz={state.current_vz:+5.2f}m/s "
        f"motors=[{motors_str}] max_m={float(np.max(state.motors)):.3f} "
        f"modes=[{arm}|{ang}|{alt}]"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    # Compute motor trend: compare early window (1s - 1s+TREND_WINDOW) vs late window (last TREND_WINDOW)
    trend_early_max = 0.0
    trend_late_max = 0.0
    trend_valid = False
    if state.motor_samples:
        early_start = TREND_IGNORE_INITIAL
        early_end = TREND_IGNORE_INITIAL + TREND_WINDOW
        late_start = GROUND_IDLE_DURATION - TREND_WINDOW

        early_samples = [m for (t, m) in state.motor_samples if early_start <= t < early_end]
        late_samples = [m for (t, m) in state.motor_samples if t >= late_start]

        if early_samples and late_samples:
            trend_early_max = max(early_samples)
            trend_late_max = max(late_samples)
            trend_valid = True

    print()
    print("=" * 70)
    print("  E2E GROUND IDLE TEST RESULTS")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print()
    print(f"  --- Ground Idle ---")
    print(f"  Max altitude:         {state.max_altitude_during_idle:.3f}m (limit: {GROUND_IDLE_MAX_ALT}m)")
    print(f"  Max motor overall:    {state.max_motor:.3f}")
    if trend_valid:
        print(f"  Motor trend (early):  {trend_early_max:.3f} (window {TREND_IGNORE_INITIAL:.0f}s-{TREND_IGNORE_INITIAL+TREND_WINDOW:.0f}s)")
        print(f"  Motor trend (late):   {trend_late_max:.3f} (last {TREND_WINDOW:.0f}s)")
        print(f"  Motor trend increase: {trend_late_max - trend_early_max:+.3f} (max allowed: +{TREND_MAX_INCREASE:.3f})")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if state.max_altitude_during_idle > GROUND_IDLE_MAX_ALT:
        passed = False
        issues.append(
            f"Max altitude during idle {state.max_altitude_during_idle:.3f}m > {GROUND_IDLE_MAX_ALT}m"
        )

    if trend_valid and (trend_late_max - trend_early_max) > TREND_MAX_INCREASE:
        passed = False
        issues.append(
            f"Motors show sustained upward trend: late={trend_late_max:.3f} vs early={trend_early_max:.3f} "
            f"(increase {trend_late_max - trend_early_max:+.3f} > {TREND_MAX_INCREASE})"
        )

    if state.max_motor >= MOTOR_SOFT_WARNING:
        issues.append(
            f"WARNING: max motor {state.max_motor:.3f} >= {MOTOR_SOFT_WARNING} "
            f"(soft threshold — may indicate thrust buildup)"
        )

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses")

    if passed and not any("CRASH" in i or "too" in i or "sustained" in i for i in issues):
        print("  Status:               PASS")
        for issue in issues:
            print(f"    {issue}")
    elif passed:
        print("  Status:               PASS (with warnings)")
        for issue in issues:
            print(f"    {issue}")
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
        hsplit name = "Ground Idle Test" {
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
    "e2e-ground-idle-test.kdl",
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
            print("  E2E GROUND IDLE Test (INITIALIZE thrust buildup regression)")
            print(f"  Scenario: ARM with ALTHOLD + low throttle, sit {GROUND_IDLE_DURATION}s, no liftoff")
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
db_path = "/tmp/e2e_ground_idle_test_db" if running_under_editor else "e2e_ground_idle_test_db"

print(f"E2E GROUND IDLE Test")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Scenario: ARM + ALTHOLD + low throttle — must not lift off")

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
