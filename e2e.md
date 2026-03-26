# E2E ALT_HOLD Flight Testing

## Goal

Automated end-to-end testing of Betaflight's ALT_HOLD flight mode using the Elodin simulation platform. Betaflight SITL runs the full flight controller, Elodin runs physics simulation, communication via UDP lockstep at 1kHz.

## Current Status (2026-03-27): PASSING

```
  Target altitude:      7m
  Max altitude:         7.0m
  Hover target:         6.9m
  Hover max drift:      0.5m       (within 1.0m tolerance)
  Hover avg altitude:   6.6m
  Landing altitude:     0.02m      (contact-based disarm)
  Landing velocity:     0.00m/s
  Sim speed:            0.9x realtime
  Status:               PASS
```

## Build & Run

### Build Betaflight SITL
```bash
cd betaflight/
make arm_sdk_install        # first time only
make TARGET=SITL            # incremental build
# or
make TARGET=SITL clean && make TARGET=SITL   # full rebuild
```

### Build Elodin (requires nix)
```bash
cd elodin/
nix develop
source $NIX_SHELLRC
./run.sh rebuild-elodin     # builds Python SDK + editor binary
```

### Run E2E Test
```bash
cd elodin/
./run.sh e2e-althold 2>&1 | tee /tmp/bf-e2e-all.log       # headless
./run.sh e2e-althold-editor 2>&1 | tee /tmp/bf-e2e-all.log # with 3D viewport
```

## Eeprom Configuration

Required before first run. Settings persist in `betaflight/eeprom.bin` across rebuilds. Delete to reset.

```bash
# Terminal 1: Start SITL
cd betaflight/ && ./obj/main/betaflight_SITL.elf

# Terminal 2: Connect CLI
socat -,rawer TCP:127.0.0.1:5761
# Type # then Enter, then:

set small_angle = 180
set acc_calibration = 0,0,0,1
set ap_hover_throttle = 1130
set failsafe_delay = 200
set d_pitch = 0
set d_roll = 0
aux 0 0 0 1700 2100 0 0
aux 1 1 1 1700 2100 0 0
aux 2 3 2 1700 2100 0 0
save
```

**Aux channel mapping:**
```
aux <index> <permanentId> <auxChannelIndex> <start> <end> <logic> <linkedTo>

aux 0  0  0  1700 2100 0 0   ->  BOXARM      on AUX1 (rcData[4])
aux 1  1  1  1700 2100 0 0   ->  BOXANGLE    on AUX2 (rcData[5])
aux 2  3  2  1700 2100 0 0   ->  BOXALTHOLD  on AUX3 (rcData[6])
```

**Notes:**
- `d_pitch=0` and `d_roll=0` required for SITL (D-term causes liftoff oscillation)
- Use `socat -,rawer` for CLI (screen/PTY method may hang)
- Type `exit` to leave CLI (reboots SITL)

## Test Phases

```
Phase            Throttle  ARM   ANGLE  ALTHOLD   Transition Condition
----------------------------------------------------------------------
BOOT             1000      off   off    off       5s elapsed (gyro cal)
ARM              1000      ON    ON     off       2s elapsed
ENABLE_ALTHOLD   1000      ON    ON     ON        2s elapsed (takeoff prep)
SETTLE           1500      ON    ON     ON        1s (enables stick adjust)
CLIMB            1700      ON    ON     ON        altitude >= 5.5m
TOP_APPROACH     1600      ON    ON     ON        altitude >= 6.8m AND |vz| < 0.2 for 0.5s
HOVER            1500      ON    ON     ON        10s elapsed
DESCEND          1300      ON    ON     ON        altitude <= 2m (or 60s)
APPROACH         1380      ON    ON     ON        altitude <= 0.5m
LAND             1380      ON    ON     ON        contact-based: alt < 0.1m AND |vz| < 0.2 for 0.5s
DISARM           1000      off   off    off       1s elapsed
DONE             --        --    --     --        print results
```

**Key phase details:**
- **TOP_APPROACH**: Slow climb (throttle 1600, above deadband) lets BF estimator converge before hover. This is a harness mitigation for estimator lag, not an estimator fix.
- **LAND**: Contact-based disarm replaces simple altitude threshold. Requires low altitude AND low velocity held for 0.5s.
- **LOW_HOVER**: Removed — ground effect is disabled, so near-ground hold is not meaningful yet. Will return with sim realism package.

## ALT_HOLD Architecture

### Controller (iNav-style cascaded)
```
RC stick -> [Stick Deadband +-50] -> [Sqrt Controller] -> Target velocity
         -> [Acceleration Limiter (0.5G up, 0.8G down)]
         -> [Velocity PID with back-calculation anti-windup]
         -> [PT1 Filter 4Hz] -> hover_throttle + correction -> mixer
```

### Altitude Estimator (position.c)
```
Fast predictor (1kHz, PID loop):
  accel -> rMat rotation -> earth-frame Z -> integrate velocity & altitude

Slow corrector (~50Hz, TASK_ALTITUDE):
  baro -> 1Hz LPF -> residual corrections to velocity, altitude & accel bias
  Weights: est_w_z_baro_p=0.35, est_w_z_baro_v=0.35, est_w_acc_bias=0.01
  + velocity residual decay when no baro data
```

**Estimator lag:** ~0.6m during climb/descent transitions. Converges to near-zero residual over ~10s at steady state. This is why TOP_APPROACH exists.

## Key Code Changes

### BOXALTHOLD Mode Activation (core.c)
ALT_HOLD is activated via BF's standard mode system: `IS_RC_MODE_ACTIVE(BOXALTHOLD)`. Requires eeprom `aux 2 3 2 1700 2100 0 0` (permanentId=3). The arming safety check at core.c:330 prevents arming with ALTHOLD switch already on.

### Altitude Estimator (position.c)
Two-layer complementary filter ported from iNav. Key fix: removed `WAS_EVER_ARMED` gate that blocked velocity integration on first flight.

### D-term Workaround
BF's rate PID D-term (Kd=34) amplifies gyro transients at liftoff in the sim. The ground contact model creates a sudden gyro spike at ground-to-air transition. Setting `d_roll=0 d_pitch=0` eliminates the oscillation. Sim-specific workaround, not needed for real hardware.

### Viewport Fix (libs/db/src/lib.rs)
`ctx.truncate()` after warmup caused VTable rebuild with 0 fields, preventing 3D model updates. Fixed by skipping empty VTable commits.

## Debug Instrumentation

All debug prints use `fprintf(stderr, ...)` because SITL stdout is buffered and lost when killed. **Warning:** heavy debug prints in the PID loop cause timing changes that affect test results.

| Tag | File | Status | What It Shows |
|-----|------|--------|--------------|
| `[MIXER]` | mixer.c | **Active** (~1Hz) | Throttle before/after ALT_HOLD, mixRange, motor values |
| `[PREDICT]` | position.c | Commented | Predictor: accel, velocity, altitude, dt |
| `[ATTR]` | position.c | Commented | Attribution: cumulative pred/baro/decay velocity |
| `[ALTHOLD]` | autopilot_multirotor.c | Commented | Controller: throttle, target/actual alt/vel, PID correction |
| `[ATTITUDE]` | autopilot_multirotor.c | Commented | Attitude: roll/pitch, PID outputs, gyro rates |
| `[PITCH]`/`[ROLL]` | pid.c | Commented | PID split: setAngle, curAngle, P/I/D/F/Sum |

## Known Limitations

1. **D-term disabled** — `d_roll=0 d_pitch=0` required for SITL
2. **Ground effect zeroed** — baro_bias=0, force_std=0, torque_std=0 in config.py
3. **Estimator lag** — BF altitude lags ~0.6m during transitions (converges at steady state). Hover target captures at ~6.25m when truth is ~6.9m; hover avg ~6.6m
4. **Landing approach speed** — APPROACH->LAND at ~0.5m with vz~-0.73m/s. Touchdown is clean but flare would be more realistic

## Full Progress

See `betaflight/progress.md` for complete bug fix timeline and future work priorities.
