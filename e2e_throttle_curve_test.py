#!/usr/bin/env python3
"""
E2E Throttle Curve Test — validates curve-aware center-stick semantics.

MANUAL TEST — not in automation gate. Requires custom EEPROM configuration.

Validates that the curve-aware rcLookupThrottleMid() produces the correct
zero point even with non-default throttle curve settings. With default
linear curve, a wrong implementation could still pass.

Prerequisites — MUST configure EEPROM before running:
    # Connect to SITL CLI:
    socat PTY,link=/tmp/bf-cli,rawer TCP:127.0.0.1:5761 &
    screen /tmp/bf-cli
    # Type '#' + Enter, then:
    set thr_mid = 30
    set thr_expo = 40
    save

    # After test, RESTORE defaults:
    set thr_mid = 50
    set thr_expo = 0
    save

Also requires standard ALTHOLD eeprom config (aux, failsafe, etc.).
Assumes default midrc = 1500 and deadband = 50.

Flight phases:
  BOOT(5s) → PRESELECT(2s) → ARM(throttle LOW, 2s)
  → BELOW_MID(midrc-300=1200, 5s — must NOT climb)
  → CLIMB(midrc+200=1700, wait alt > 2m)
  → HOLD(midrc=1500, 5s — drift < 1.0m)
  → DISARM → DONE

Pass criteria:
  - BELOW_MID: z < 0.1m (no liftoff)
  - CLIMB: alt > 2m within 30s
  - HOLD: drift < 1.0m from capture altitude

How to run (manual, NOT via run.sh):
    # 1. Configure EEPROM with thr_mid=30, thr_expo=40 (see above)
    # 2. Start SITL: cd betaflight && ./obj/main/betaflight_SITL.elf
    # 3. Run test:
    cd elodin && python3 e2e_throttle_curve_test.py run --no-s10
    # 4. Restore EEPROM: set thr_mid=50, thr_expo=0, save
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

MIDRC = 1500
DEADBAND = 50

STICK_BELOW_MID = MIDRC - 300     # 1200
STICK_CLIMB = MIDRC + 200         # 1700
STICK_HOLD = MIDRC                # 1500

BELOW_MID_DURATION = 5.0
CLIMB_TIMEOUT = 30.0
CLIMB_TARGET_ALT = 2.0
HOLD_DURATION = 5.0

GROUND_MAX_ALT = 0.1
HOLD_MAX_DRIFT = 1.0

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
ALTHOLD_SETTLE = 2.0
DISARM_DURATION = 1.0


# ============================================================================
#  PHASE STATE MACHINE
# ============================================================================

class Phase(Enum):
    BOOT = auto()
    PRESELECT = auto()
    ARM = auto()
    BELOW_MID = auto()
    CLIMB = auto()
    HOLD = auto()
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

    max_alt_below_mid: float = 0.0
    hold_capture_alt: float = 0.0
    hold_max_drift: float = 0.0

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

    elif phase == Phase.BELOW_MID:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_BELOW_MID

    elif phase == Phase.CLIMB:
        channels[CH_ARM] = MODE_ON
        channels[CH_ANGLE] = MODE_ON
        channels[CH_ALTHOLD] = MODE_ON
        channels[CH_THROTTLE] = STICK_CLIMB

    elif phase == Phase.HOLD:
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

    if phase == Phase.BOOT:
        if elapsed >= BOOT_DURATION:
            state.transition(Phase.PRESELECT, t)

    elif phase == Phase.PRESELECT:
        if elapsed >= ALTHOLD_SETTLE:
            state.transition(Phase.ARM, t)

    elif phase == Phase.ARM:
        if elapsed >= ARM_DURATION:
            print(f"[{t:6.1f}s] [           ARM] Testing below-mid with non-default curve ({STICK_BELOW_MID})...")
            state.transition(Phase.BELOW_MID, t)

    elif phase == Phase.BELOW_MID:
        state.max_alt_below_mid = max(state.max_alt_below_mid, state.current_altitude)
        if state.current_altitude > GROUND_MAX_ALT:
            state.crash_detected = True
            state.crash_reason = f"Liftoff during BELOW_MID with custom curve (alt={state.current_altitude:.3f}m)"
            state.transition(Phase.DISARM, t)
            return
        if elapsed >= BELOW_MID_DURATION:
            print(f"[{t:6.1f}s] [     BELOW_MID] OK. Climbing ({STICK_CLIMB})...")
            state.transition(Phase.CLIMB, t)

    elif phase == Phase.CLIMB:
        if state.current_altitude > CLIMB_TARGET_ALT:
            state.hold_capture_alt = state.current_altitude
            print(f"[{t:6.1f}s] [         CLIMB] Reached {state.current_altitude:.1f}m. Hold check...")
            state.transition(Phase.HOLD, t)
        elif elapsed >= CLIMB_TIMEOUT:
            state.crash_detected = True
            state.crash_reason = f"CLIMB timeout with custom curve: alt={state.current_altitude:.3f}m < {CLIMB_TARGET_ALT}m"
            state.transition(Phase.DISARM, t)

    elif phase == Phase.HOLD:
        drift = abs(state.current_altitude - state.hold_capture_alt)
        state.hold_max_drift = max(state.hold_max_drift, drift)
        if elapsed >= HOLD_DURATION:
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

    print(
        f"[{t:6.1f}s] [{state.phase.name:>14}] "
        f"alt={state.current_altitude:+7.3f}m vz={state.current_vz:+5.2f}m/s "
        f"T={thr} motors=[{motors_str}]"
    )


def print_results(state: TestState):
    if state.results_printed:
        return
    state.results_printed = True

    print()
    print("=" * 70)
    print("  E2E THROTTLE CURVE TEST RESULTS (manual, non-default curve)")
    print("  NOTE: thr_mid=30, thr_expo=40 must be set in EEPROM before running")
    print("=" * 70)
    print(f"  Duration:             {state.sim_time:.1f}s")
    print(f"  Lockstep steps:       {state.step_count}")
    print()
    print(f"  BELOW_MID max alt:    {state.max_alt_below_mid:.3f}m (limit={GROUND_MAX_ALT}m)")
    print(f"  Hold capture alt:     {state.hold_capture_alt:.1f}m")
    print(f"  Hold max drift:       {state.hold_max_drift:.2f}m (limit={HOLD_MAX_DRIFT}m)")
    if state.crash_detected:
        print(f"  Crash:                {state.crash_reason}")
    print()

    passed = True
    issues = []

    if state.crash_detected:
        passed = False
        issues.append(f"CRASH: {state.crash_reason}")

    if state.max_alt_below_mid > GROUND_MAX_ALT:
        passed = False
        issues.append(f"BELOW_MID liftoff with custom curve: {state.max_alt_below_mid:.3f}m > {GROUND_MAX_ALT}m")

    if state.hold_max_drift > HOLD_MAX_DRIFT:
        passed = False
        issues.append(f"Hold drift with custom curve: {state.hold_max_drift:.2f}m > {HOLD_MAX_DRIFT}m")

    if state.step_count == 0:
        passed = False
        issues.append("No motor responses")

    print(f"  Status:               {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        print(f"    {'FAIL' if not passed else 'INFO'}: {issue}")
    print()
    print("  REMINDER: Restore EEPROM after test:")
    print("    set thr_mid = 50")
    print("    set thr_expo = 0")
    print("    save")
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
        hsplit name = "Throttle Curve Test" {
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
    "e2e-throttle-curve-test.kdl",
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
            print("  E2E THROTTLE CURVE Test (manual, non-default curve)")
            print(f"  Requires EEPROM: thr_mid=30, thr_expo=40")
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
db_path = "/tmp/e2e_throttle_curve_test_db" if running_under_editor else "e2e_throttle_curve_test_db"

print(f"E2E THROTTLE CURVE Test (manual)")
print(f"  SITL: {BETAFLIGHT_PATH.name}")
print(f"  Requires: thr_mid=30, thr_expo=40 in EEPROM")

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
