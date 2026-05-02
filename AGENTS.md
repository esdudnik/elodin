# Elodin

Elodin is an open-source platform for rapid design, testing, and simulation of aerospace and physical systems — aerospace's answer to ROS. This monorepo contains the Elodin Editor (3D viewer/graphing), Elodin DB (time-series telemetry database), the Python SDK (nox-py, JAX-based simulation), flight software components, and the Aleph flight computer NixOS configuration (NVIDIA Jetson Orin).

## Rules

- Always use the `nix develop` shell when developing changes.
- Always use `uv` inside the nix shell for Python everything (the `install-elodin` shell function sets this up).
- Don't commit changes to git — that's for the developer to do
- When suggesting new dependencies, check they are well supported and maintained.
- Never use unsafe Rust code.
- Run all activities from the repository root.

## Quick Start

### Python SDK (always build first, so binaries pick it up)

```bash
nix develop
install-elodin
```

### CI Checks

```bash
cargo fmt && cargo test && cargo clippy -- -Dwarnings   # Rust
ruff format --check && ruff check --fix                  # Python
alejandra                                                # Nix
```

## Detailed Guidance

For in-depth instructions, read the relevant skill file below when working in that area:

- **Creating simulations** (Python SDK, components, systems, 6DOF, visualization, SITL/HITL): `.cursor/skills/elodin-simulation/SKILL.md`
- **Contributing to the codebase** (building from source, architecture, workspace structure, testing): `.cursor/skills/elodin-dev/SKILL.md`
- **Aleph flight computer** (AlephOS deployment, NixOS modules, flight software services, firmware): `.cursor/skills/elodin-aleph/SKILL.md`
- **Elodin DB** (running the database, client integrations, replication/follow mode, Lua REPL): `.cursor/skills/elodin-db/SKILL.md`
- **Elodin Editor** (Bevy/Egui architecture, hot-reload, viewport, telemetry graphs, KDL schematics): `.cursor/skills/elodin-editor-dev/SKILL.md`
- **Python SDK internals** (PyO3 bindings, nox-py, JAX integration, adding components/systems): `.cursor/skills/nox-py-dev/SKILL.md`
- **Nix environment** (dev shell troubleshooting, OrbStack VMs, flake.nix, binary cache): `.cursor/skills/elodin-nix/SKILL.md`
- **IREE runtime** (iree-runtime crate, VMFB execution from Rust, FFI bindings, dual-backend architecture): `.cursor/skills/elodin-iree/SKILL.md`

## Architecture

- **Workspace:** 57 Cargo crates with `cargo elodin` / `cargo elodin-db` aliases
- **Dual execution backends:** IREE (default, fast, no GIL) and JAX (fallback, full compatibility)
- **Impeller2:** zero-copy pub-sub protocol for telemetry (FNV-1a hashed component IDs)
- **Editor:** Bevy + Egui with modular plugin architecture and KDL schematic hot-reload
- **s10:** process orchestrator -- manages sim subprocess and external processes (e.g. Betaflight SITL)
- **Stellarator:** deterministic single-threaded async runtime for flight software
- **Roci:** composable flight software framework with pipe-based system composition
- Rust edition 2024, toolchain 1.90.0

## Running Betaflight SITL Simulation

The `run.sh` script in this directory handles building and running the elodin editor with the betaflight-sitl example. Must be run from the elodin directory root.

### Prerequisites (one-time)
```bash
nix develop                    # Enter nix shell (required)
source $NIX_SHELLRC            # Load shell functions
install-elodin                 # Build everything (Python SDK + binaries)
```

### Using run.sh

```bash
./run.sh build-bf                        # Clean + build betaflight SITL .elf
./run.sh rebuild-elodin                  # Rebuild elodin Python SDK + editor binary
./run.sh run                             # Run editor with betaflight-sitl example (skip rebuild)
./run.sh all                             # build-bf + rebuild-elodin + run (default)
./run.sh check                           # Analyze log file at /tmp/bf-elodin.log

# Single E2E tests (production physics: baseline IGE gates)
./run.sh e2e-ground-idle                 # ground-idle regression test
./run.sh e2e-failsafe-althold            # failsafe descent landing test
./run.sh e2e-all                         # 11-test regression suite (~25 min)
# (and many more — see ./run.sh with no args)

# Diagnostic physics profiles (since 2026-05-01)
./run.sh e2e-strict    [N|--runs=N]      # focused suite under "strict" IGE profile (partial gate)
./run.sh e2e-realistic [N|--runs=N]      # focused suite under "realistic" IGE profile (no gate, matches real HW)
```

The `e2e-strict` and `e2e-realistic` targets set `E2E_PHYSICS_PROFILE` and run
the focused subset (`ground-idle`, `nosettle-takeoff`, `failsafe-althold`,
`failsafe-init`, `midair-activation`) — used to expose IGE-related controller
or mixer issues that would otherwise be hidden by the `baseline` thrust gate.
Severity ordering: `baseline` < `strict` < `realistic`.
See `sim_world.md` for the physics-profile reference table.

### E2E command cheat-sheet (which command for which goal)

| What you want | Command | Tests run | Profile | Cycles | Approx duration |
|---|---|---|---|---|---|
| Smoke test (focused, single cycle) | `./run.sh e2e-realistic` | Focused 5 | realistic | 1 | ~8 min |
| Confidence run (focused, multi-cycle) | `./run.sh e2e-realistic --runs=5` | Focused 5 | realistic | 5 (= 25 runs) | ~40 min |
| Stricter-than-baseline regression pressure | `./run.sh e2e-strict --runs=5` | Focused 5 | strict | 5 | ~40 min |
| Baseline regression (default CI gate) | `./run.sh e2e-all` | All 11 | baseline | 1 | ~25 min |
| All 11 tests under realistic (no built-in target) | `E2E_PHYSICS_PROFILE=realistic ./run.sh e2e-all` | All 11 | realistic | 1 | ~25 min |
| All 11, multi-run, realistic (manual loop) | `for i in 1 2 3 4 5; do E2E_PHYSICS_PROFILE=realistic ./run.sh e2e-all; done` | All 11 | realistic | 5 (= 55 runs) | ~2h |
| Single individual test, default profile | `./run.sh e2e-ground-idle` (or any other `e2e-*` name) | 1 | baseline (default) | 1 | ~1-3 min |
| Single individual test under realistic | `E2E_PHYSICS_PROFILE=realistic ./run.sh e2e-ground-idle` | 1 | realistic | 1 | ~1-3 min |

**Focused 5 vs All 11:**
- **Focused 5** (IGE-sensitive): `ground-idle`, `nosettle-takeoff`, `failsafe-althold`, `failsafe-init`, `midair-activation` — exercise near-ground / failsafe paths where IGE matters.
- **All 11** (`e2e-all` set): focused 5 + `smooth-takeoff`, `center-semantics`, `angle-althold`, `acro-althold`, `flight`, `poshold` — extra 6 are in-flight or non-IGE behavior; running them under `realistic` adds time without adding diagnostic value (per `strict_test.md`).

**Env-var trick:** `E2E_PHYSICS_PROFILE=name ./run.sh <target>` sets the profile for that one invocation only. Variable doesn't persist in your shell after the command finishes. Useful for ad-hoc combinations not covered by the dedicated targets.

**What the runner reports** (since Step 1.5, 2026-05-02): both `e2e-all` and `e2e-strict`/`e2e-realistic` summaries include `Pass / Ctrl-fail / Infra-fail` three-column counts, `Effective pass rate` excluding infra failures, per-test result lines tagged `PASS`/`FAIL`/`INFRA`, an `Infra-fail reasons:` section with causes (`ports-still-held`, `sitl-died-during-test`, `startup-timeout`, etc.), and a three-state return code (0/1/2).

**Logs:**
- Single test: `/tmp/bf-e2e-<test>.log`
- `e2e-all` suite: `/tmp/bf-e2e-e2e-<test>.log` (note the doubled `e2e` from how `e2e-all` builds names)
- Focused suite per-run: `/tmp/bf-e2e-<profile>-<test>-r<N>.log`

`rebuild-elodin` requires `nix develop` shell. `build-bf` and `run` do not.

### What run.sh does
- **build-bf**: Installs ARM SDK (if needed), cleans and builds betaflight SITL binary (`make TARGET=SITL`) from the top-level `../betaflight/` directory
- **rebuild-elodin**: Detects nix Python 3.13, creates `.venv`, builds nox-py wheel via `maturin develop`, installs elodin editor binary via `cargo install`
- **run**: Activates venv, runs `elodin editor examples/betaflight-sitl/main.py`

### Manual run (alternative)
```bash
nix develop
source $NIX_SHELLRC
source .venv/bin/activate
export PATH="$HOME/.cargo/bin:$PATH"
unset PYTHONPATH
elodin editor examples/betaflight-sitl/main.py
```

See `run.md` for the full manual setup guide.

### Environment Requirements
- `.venv` **must** use Python 3.13 (matches nix's Python for PyO3 linkage)
- `PYTHONPATH` must be unset (prevents numpy conflicts)
- `uv` for all Python package management (inside nix shell)
- `hidapi` pip package required for TX12 joystick input

## Betaflight SITL Integration

Located in `examples/betaflight-sitl/`. Key files:

| File | Purpose |
|------|---------|
| `main.py` | Entry point: spawns drone, ground entity, joystick input, runs world |
| `sim.py` | Physics: motor dynamics, 6-DOF rigid body, quadratic drag, ground collision |
| `comms.py` | FDM packet parsing, MSP protocol, UDP lockstep communication |
| `sensors.py` | Simulated IMU, barometer, airspeed sensor |
| `config.py` | Simulation parameters (time step, motor constants, drag coefficients) |

### Communication Ports
- **UDP 9002**: PWM/motor commands from Betaflight SITL
- **UDP 9003**: FDM packets (sensors) to Betaflight SITL
- **TCP 5761**: MSP protocol (Configurator, joystick)

### How it works
1. `elodin editor` reads `main.py`, generates s10.toml (process orchestration plan)
2. s10 spawns: Python sim subprocess + Betaflight SITL binary
3. Sim starts elodin-db on port 2240, editor connects via TCP
4. Physics loop: receive motor commands (UDP) -> simulate step -> send FDM sensors (UDP)
5. Joystick: TX12 via hidapi (VID=0x1209 PID=0x4F54) sends RC channels via MSP

### Current State
- Backend: `jax` (IREE does not support all JAX features used)
- Sim speed: ~0.3x realtime at 1kHz time step
- Sim time step: 0.001s (1 kHz, configured in `config.py`)
- 3D viewport: working (requires fixes in `libs/db/src/lib.rs` and `libs/impeller2/bevy/src/lib.rs`)
- Joystick: working (TX12 USB HID via hidapi)

## Reference Documentation

- **`help/changes.md`** -- All local modifications after pulling from remote, organized by category with rationale
- **`help/betaflight.md`** -- Betaflight flight controller reference: SITL target, packet structures, motor mapping, key source files
