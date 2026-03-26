# Elodin -- Local Changes After Pull

All changes are relative to `origin/main` (HEAD: `9103f38d`). No local commits -- all modifications are unstaged working tree changes.

## Summary

9 modified files, 3 new files, 1 dirty submodule. Changes fall into 6 categories:
1. Joystick input feature
2. 3D viewport fixes
3. Diagnostic logging & error handling
4. Process management fix
5. Simulation config tuning
6. Build/run tooling

---

## 1. Joystick Input Feature

**File**: `examples/betaflight-sitl/main.py`
**Why**: Enable real-time manual control of the simulated drone using a physical TX12 radio controller via USB HID, replacing the scripted arm/throttle sequence.

**Changes**:
- Added TX12 USB HID constants (VID=0x1209, PID=0x4F54) and a background thread reading joystick at 200Hz via `hidapi`
- Shared `_js_channels` numpy array (16 channels) bridges joystick thread to simulation loop
- When joystick connected: `interactive=True` mode, `max_ticks` = 1 hour, editor controls pause/resume
- When no joystick: falls back to original scripted sequence (boot -> arm -> throttle -> disarm)
- Added static `ground` entity at origin (0.001 kg) as camera orbit target
- Changed camera expression to `translate_world(5.0, 5.0, 3.0)`

---

## 2. 3D Viewport Fixes

**File**: `libs/db/src/lib.rs`
**Why**: Objects were not appearing in the 3D viewport until data started flowing. The empty-timeseries guards prevented the editor from receiving component metadata before any simulation data was written.

**Changes** (3 locations):
- `vtable()` method: removed `component.time_series.index().is_empty()` guard -- now all components are included in VTable regardless of data presence
- `visit()` loop: removed empty check when iterating for `get_nearest()` -- now processes all components
- `fill_table()` method: removed empty check when filling table with `latest()` -- now processes all components

**Effect**: Editor receives VTable entries and data for components immediately upon subscription, fixing the blank viewport on startup.

---

## 3. Diagnostic Logging & Error Handling

### 3a. Editor playback logging
**File**: `libs/elodin-editor/src/lib.rs`
**Why**: Needed visibility into playback state transitions for debugging viewport timing issues.

**Changes**: Added 3 `tracing::trace!()` calls in `advance_playback` -- logs paused state, stalled playback, and advancing with timestamps.

### 3b. 3D object diagnostics
**File**: `libs/elodin-editor/src/object_3d.rs`
**Why**: Objects sometimes failed to appear in viewport with no indication of why. The original compact `if let` chain silently skipped failures.

**Changes**: Refactored to explicit `match` arms with `warn_once!` logging for: `as_world_pos()` returning None, EQL execution failure, and missing compiled expression.

### 3c. Telemetry pipeline diagnostics
**File**: `libs/impeller2/bevy/src/lib.rs`
**Why**: Data flow from elodin-db to the Bevy renderer was opaque. Needed to trace where packets were being received, cached, and applied to diagnose missing objects.

**Changes**:
- Added `entry_count()` and `total_entries()` methods on `TelemetryCache`
- `apply_cached_data`: frame counter, applied/skipped tracking, trace summary per frame, per-component cache depth every 100 frames
- `sink_inner`: packet-level trace logging (VTableMsg fields, timestamps, table counts)
- **Behavioral fix**: `table.sink()` errors are now logged as warnings instead of silently ignored (`let _ =`)

---

## 4. Process Management Fix

**File**: `libs/s10/src/sim.rs`
**Why**: Spawned child processes (like Betaflight SITL) would receive SIGTTIN and stop when they tried to read from stdin. This happened because `process_group(0)` puts children in a background process group on macOS, and reading from stdin in a background group triggers SIGTTIN.

**Changes**: Added `cmd.stdin(std::process::Stdio::null())` to detach stdin from child processes.

---

## 5. Simulation Config Tuning

**File**: `examples/betaflight-sitl/config.py`
**Why**: 4kHz (0.000250s) simulation rate was too slow in practice (~0.075x realtime). Reducing to 1kHz (0.001s) improved sim speed to ~0.3x realtime while maintaining acceptable fidelity.

**Changes**: `sim_time_step` changed from `0.000250` (4kHz) to `0.001000` (1kHz). Both values kept as comments for easy switching.

---

## 6. Build/Run Tooling

### 6a. Build/run script
**File**: `run.sh` (new, 209 lines)
**Why**: Manual setup was error-prone (nix python detection, venv creation, env vars, build steps). Automates the full build-and-run workflow.

**Modes**: `rebuild` (maturin + cargo install), `run` (activate venv + launch editor), `all` (both), `check` (analyze log file).

### 6b. Manual setup guide
**File**: `run.md` (new, 52 lines)
**Why**: Documents the manual alternative to run.sh for when the script isn't suitable.

### 6c. WebSocket proxy
**File**: `examples/betaflight-sitl/ws-proxy.py` (new, 93 lines)
**Why**: Betaflight Configurator PWA uses WebSocket, but SITL exposes MSP on raw TCP (port 5761). This proxy bridges `ws://127.0.0.1:5762` to `tcp://127.0.0.1:5761`.

### 6d. Agent instructions
**File**: `AGENTS.md`
**Why**: Mirrors CLAUDE.md content for AI agents that use AGENTS.md convention instead of CLAUDE.md.

---

## 7. Betaflight Path Change

**File**: `examples/betaflight-sitl/main.py`
**Why**: Use the top-level betaflight fork (`../betaflight/`) instead of the git submodule for SITL binary and eeprom.bin. Allows single-copy development workflow — build betaflight once, use from elodin.

**Changes**:
- `BETAFLIGHT_PATH` now points to `../betaflight/obj/main/betaflight_SITL.elf` (was `betaflight/obj/main/...` submodule)
- `cwd` in s10 recipe now points to `../betaflight/` (was example directory)
- `run.sh`: renamed `rebuild` to `rebuild-elodin`, added `build-bf` command for betaflight SITL builds

---

## Submodule

**File**: `examples/betaflight-sitl/betaflight`
**Status**: Submodule still exists but is no longer used by `main.py`. The SITL binary and eeprom.bin are now sourced from the top-level `../betaflight/` directory.
