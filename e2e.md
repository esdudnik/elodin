# E2E ALT_HOLD Flight Testing

## Goal

Automated end-to-end testing of Betaflight's iNav-style ALT_HOLD flight mode using
the Elodin simulation platform. Betaflight SITL runs the full flight controller,
Elodin runs physics simulation, communication via UDP lockstep at 1 kHz.

## Current Status (2026-05-13): 100% pass, 0 infra across post-mutex runs

After the dyad thread-safety mutex landed (`a5cb52222` in betaflight, 2026-05-13),
all categories of failure went to zero:

| Suite | Profile | Runs | Pass | Ctrl-fail | Final infra |
|---|---|---|---|---|---|
| `e2e-realistic --runs=10` (post-mutex) | realistic | 50 | **50** | **0** | **0** |
| `e2e-all --runs=5` (post-mutex) | baseline | 55 | **55** | **0** | **0** |
| **Combined post-mutex** | — | **105** | **105** | **0** | **0** |
| **Cumulative all post-fix runs** | — | **260+** | — | **0** | resolved by Session 15 |

Hardware validation pending. See `betaflight/progress.md` Session 14-15 for full
context.

## Test Suite (12 tests, 11 automated + 1 manual)

| # | Test | File | Command | What it validates |
|---|------|------|---------|-------------------|
| 1 | Ground idle | `e2e_ground_idle_test.py` | `./run.sh e2e-ground-idle` | No moon-takeoff at idle stick |
| 2 | Smooth takeoff | `e2e_smooth_takeoff_test.py` | `./run.sh e2e-smooth-takeoff` | Liftoff only above center+deadband |
| 3 | Center semantics | `e2e_center_semantics_test.py` | `./run.sh e2e-center-semantics` | Below-mid no climb, hold, descent |
| 4 | No-settle takeoff | `e2e_nosettle_takeoff_test.py` | `./run.sh e2e-nosettle-takeoff` | Climb without throttle settling |
| 5 | ANGLE+ALTHOLD | `e2e_angle_althold_test.py` | `./run.sh e2e-angle-althold` | Full flight cycle with self-leveling |
| 6 | ACRO+ALTHOLD | `e2e_acro_althold_test.py` | `./run.sh e2e-acro-althold` | Altitude hold without ANGLE |
| 7 | Horizontal flight | `e2e_flight_test.py` | `./run.sh e2e-flight` | Lateral maneuvers + altitude hold |
| 8 | Failsafe ALTHOLD | `e2e_failsafe_althold_test.py` | `./run.sh e2e-failsafe-althold` | BOXFAILSAFE landing from hover |
| 9 | Failsafe init | `e2e_failsafe_initialize_test.py` | `./run.sh e2e-failsafe-init` | Failsafe from INITIALIZE state |
| 10 | Mid-air activation | `e2e_midair_activation_test.py` | `./run.sh e2e-midair-activation` | Safe activation at low stick mid-flight |
| 11 | POSHOLD | `e2e_poshold_test.py` | `./run.sh e2e-poshold` | Position hold with virtual mag |
| 12 | Manual landing safety | `e2e_manual_landing_safety_test.py` | `./run.sh e2e-manual-landing-safety` | Low-pass + commit-land anti-regression |
| — | Throttle curve (manual) | `e2e_throttle_curve_test.py` | `python3 e2e_throttle_curve_test.py run --no-s10` | Curve-aware midpoint with thr_mid=30 |

All tests support `[N|--runs=N]` for multi-run with per-run logs:
```bash
./run.sh e2e-midair-activation --runs=10
./run.sh e2e-all --runs=5     # 11 tests × 5 cycles = 55 runs
```

## Physics Profiles

Selectable via `E2E_PHYSICS_PROFILE` env (default `baseline`):

| Profile | IGE thrust gate | IGE AGL gate | Use case |
|---|---|---|---|
| `baseline` | smoothstep 0.65→1.0 | smoothstep 0.03→0.08 m | CI default, regression gate |
| `strict` | smoothstep 0.3→0.6 | smoothstep 0.01→0.03 m | Long-term regression pressure |
| `realistic` | **none** (always 1.0) | smoothstep 0.01→0.03 m | Matches real-hardware physics |

Severity ordering: `baseline` < `strict` < `realistic`.

**v10.4 default**: `realistic` is the default profile for all built-in targets.
`baseline` remains a valid value for ad-hoc diagnostic override but no built-in
target uses it (tests without IGE don't model real propwash + ground effect).

**Focused suite** (6 IGE-sensitive tests: ground-idle, nosettle-takeoff,
failsafe-althold, failsafe-init, midair-activation, low-alt-horizontal):
```bash
./run.sh e2e-focused [N|--runs=N]                 # realistic + 5 cycles default
E2E_PHYSICS_PROFILE=strict ./run.sh e2e-focused   # same suite under strict
```

**Full suite under strict** (12 tests — diagnostic lane, may show new failures):
```bash
./run.sh e2e-all-strict [N|--runs=N]
```

## Build & Run

### Build Betaflight SITL
```bash
cd betaflight/
make arm_sdk_install                                       # first time only
make TARGET=SITL                                           # incremental
make TARGET=SITL clean && make TARGET=SITL                 # full rebuild
make TARGET=SITL OPTIONS="DEBUG_ALTHOLD_TRACE"             # diagnostic traces
```

### Build Elodin (requires nix)
```bash
cd elodin/
nix develop
source $NIX_SHELLRC
./run.sh rebuild-elodin
```

### Run a single test
All targets default to realistic IGE physics.
```bash
cd elodin/
./run.sh e2e-ground-idle                                                  # realistic, 1 run
./run.sh e2e-ground-idle --runs=10                                        # realistic, 10 runs
E2E_PHYSICS_PROFILE=strict ./run.sh e2e-midair-activation --runs=10       # strict override
```

### Run full suite
```bash
./run.sh e2e-all                                          # 12 tests, realistic, 1 cycle (~25 min)
./run.sh e2e-all --runs=5                                 # 12 tests × 5 cycles = 60 runs (~125 min)
./run.sh e2e-all-strict                                   # 12 tests under strict (diagnostic lane)
./run.sh e2e-focused --runs=10                            # focused 6 × 10 = 60 runs (~115 min)
```

### Editor mode (3D viewport)
Append `-editor` to any single-test command:
```bash
./run.sh e2e-angle-althold-editor
./run.sh e2e-poshold-editor
```

## Eeprom Configuration

Settings persist in `betaflight/eeprom.bin` across rebuilds. Delete to reset.

```
Terminal 1: cd betaflight/ && ./obj/main/betaflight_SITL.elf
Terminal 2: socat PTY,link=/tmp/bf-cli,rawer TCP:127.0.0.1:5761 &
            screen /tmp/bf-cli
            (type # then Enter)
```

**DEFAULT (GPS+baro + failsafe + POSHOLD)** — standard config for all tests:
```
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
```

**Aux mapping**: AUX1=ARM, AUX2=ANGLE, AUX3=ALTHOLD, AUX4=BOXFAILSAFE, AUX5=BOXPOSHOLD.

**Notes:**
- Do NOT use Betaflight Configurator — auto-dump crashes 4.6 SITL on macOS
- `socat -,rawer` may hang — use the PTY+screen pattern above
- Type `exit` or `Ctrl+D` to leave CLI (reboots SITL)
- Check `aux`, `get d_roll`, `get ap_hover_throttle` to verify

## Runner Behavior (`run.sh`)

### Three-state classification
Each test run returns one of:
- **PASS** (RC 0) — test passed
- **CONTROLLER_FAIL** (RC 1) — controller produced a wrong outcome
- **INFRA_FAIL** (RC 2) — SITL didn't start cleanly or crashed

### Retry-once on infra startup races
Infra fails matching these signatures get one retry:
- `sitl-died-no-output`
- `sitl-exited-during-startup`
- `startup-timeout`
- `sitl-alive-but-unresponsive` / `sitl-unresponsive`
- `fatal-init-signature: bind port`* (TIME_WAIT slip-through)

Other fatal signatures (Segmentation fault, Bus error, Trace/BPT trap) are
hard fails — never silently retried.

### Port hardening
`wait_ports_free` probes both `lsof` (for live PIDs) and `netstat -an -p tcp`
(for kernel-held TIME_WAIT entries on TCP ports). 30 s default timeout
accommodates macOS TIME_WAIT (2×MSL = 30 s).

### Multi-run logs
- Single run: `/tmp/bf-e2e-<test>.log`
- Multi-run with profile: `/tmp/bf-e2e-<profile>-<test>-r<N>.log`
- Multi-run e2e-all: `/tmp/bf-e2e-<test>-r<N>.log`
- Retry: `<basename>-attempt2.log`

### Summary output
Both `do_focused_suite` and `do_e2e_all` show three columns (pass / ctrl-fail /
infra-final), effective pass rate excluding infra, per-test breakdown for
multi-run, and an `Infra-fail reasons:` section. Three-state return code (0/1/2).

## ALT_HOLD Architecture

iNav-style cascaded controller (matches `navigation_multicopter.c`):

```
RC stick → [Stick Deadband ±50] → [Sqrt Controller] → Target velocity
         → [Acceleration Limiter (0.5G up, 0.8G down)]
         → [Velocity PID with Astrom back-calculation anti-windup]
         → [PT1 Filter 4Hz] → hover_throttle + correction → mixer
```

**Explicit FSM:** IDLE → INITIALIZE → IN_PROGRESS → EXITING

**Real-elapsed-time `dt`:** matches iNav's `US2S(deltaMicros)` — derived per-tick
from `currentTimeUs`, fall back to nominal `1/ALTHOLD_TASK_RATE_HZ` on first call
or >200 ms gaps. (Fix: 2026-05-06 — previous hardcoded 0.01 s was wrong against
SITL's actual ~30 ms cadence, made limiter 3× too slow.)

## Debug Instrumentation

Compile-gated traces — enabled with `OPTIONS="DEBUG_ALTHOLD_TRACE"`:

| Tag | Source | What it shows |
|-----|--------|--------------|
| `[ALT]` | `alt_hold_multirotor.c` | Tick counter, FSM state, RC values, target velocity, FSM dwell |
| `[CTRL]` | `autopilot_multirotor.c` | Target alt, desired velocity, vz, integrator, correction, throttle |
| `[LAND]` | `alt_hold_multirotor.c` | Landing detector reasons, vz/gyro gates, hold-ms |
| `[MSTOP]` | `alt_hold_multirotor.c` + `mixer.c` | Motor-stop predicate inputs and decisions |
| `[EST]` | `position.c` | Baro/GPS estimator: residuals, weights, fade |

All use `fprintf(stderr, ...)` (SITL stdout is buffered and lost on kill).
Production builds without the flag are bit-for-bit unaffected.

## Known Limitations

1. **D-term at 5/5 in SITL** — full hardware values (30/34) destabilize liftoff in sim
2. **Estimator lag** — BF altitude lags ~0.2-0.6 m during transitions, converges over ~5 s
3. **Ground effect** — IGE/VRS/wake-turbulence model is intentionally less aggressive than worst real-hardware case (`baseline` profile); use `realistic` for harder pressure
4. **SITL startup flakiness** — ~7-10% of runs hit `sitl-died-no-output` after retry. Pure infra issue, separate investigation
5. **Stale eeprom** — accumulated config state causes regressions; always reset before baseline testing

## References

- `betaflight/progress.md` — full development timeline (Sessions 1-14)
- `inav_comparison.md` — iNav vs BF architecture comparison
- `strict_test.md` — physics-profile design + investigation outcomes
- `sim_world.md` — sim physics reference (rotor aero, contact, sensors)
- `communication.md` — IPC architecture (UDP lockstep + s10)
- `betaflight/help/changes.md` § 13 — iNav ALT_HOLD as a fork feature
