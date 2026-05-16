#!/usr/bin/env bash
set -euo pipefail

# ── Elodin + Betaflight build & run script ──
# Run from the elodin repo root, inside `nix develop` shell.
# Usage: ./run.sh [build-bf|rebuild-elodin|run|e2e-all|e2e-angle-althold|...|all|check]
#   build-bf                  - clean + build betaflight SITL .elf
#   rebuild-elodin            - rebuild elodin (Python SDK + editor binary)
#   run                       - just run the editor (skip rebuild)
#   e2e-all                   - run ALL automated E2E tests sequentially
#   e2e-angle-althold         - run ANGLE+ALTHOLD E2E test (headless)
#   e2e-*-editor              - run any E2E test with 3D viewport
#   all                       - build-bf + rebuild-elodin + run (default)
#   check                     - analyze log file

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EXAMPLE="examples/betaflight-sitl/main.py"
E2E_SCRIPT="$REPO_ROOT/e2e_angle_althold_test.py"
E2E_ACRO_SCRIPT="$REPO_ROOT/e2e_acro_althold_test.py"
E2E_FLIGHT_SCRIPT="$REPO_ROOT/e2e_flight_test.py"
E2E_FAILSAFE_SCRIPT="$REPO_ROOT/e2e_failsafe_althold_test.py"
E2E_POSHOLD_SCRIPT="$REPO_ROOT/e2e_poshold_test.py"
E2E_NOSETTLE_SCRIPT="$REPO_ROOT/e2e_nosettle_takeoff_test.py"
E2E_FAILSAFE_INIT_SCRIPT="$REPO_ROOT/e2e_failsafe_initialize_test.py"
E2E_GROUND_IDLE_SCRIPT="$REPO_ROOT/e2e_ground_idle_test.py"
E2E_CENTER_SCRIPT="$REPO_ROOT/e2e_center_semantics_test.py"
E2E_MIDAIR_SCRIPT="$REPO_ROOT/e2e_midair_activation_test.py"
E2E_SMOOTH_TAKEOFF_SCRIPT="$REPO_ROOT/e2e_smooth_takeoff_test.py"
E2E_MANUAL_LANDING_SAFETY_SCRIPT="$REPO_ROOT/e2e_manual_landing_safety_test.py"
E2E_LOW_ALT_HORIZONTAL_SCRIPT="$REPO_ROOT/e2e_low_alt_horizontal_test.py"
BETAFLIGHT_DIR="$REPO_ROOT/../betaflight"
LOG_FILE="/tmp/bf-elodin.log"
E2E_LOG_FILE="/tmp/bf-e2e.log"

# v10.4: realistic IGE is the default physics profile. Tests without IGE
# (baseline profile) don't model real hardware physics (no propwash, no
# ground effect) — they're meaningless for production validation. baseline
# remains a valid profile value for ad-hoc diagnostic comparison via env
# var override, but no built-in target uses it.
export E2E_PHYSICS_PROFILE="${E2E_PHYSICS_PROFILE:-realistic}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}▶ STEP $1: $2${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail() { echo -e "${RED}  ✗ FAILED: $1${NC}"; exit 1; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }

# Detect nix Python 3.13 from the nix shell environment (no hardcoded paths)
detect_nix_python() {
    # First: prefer nix store Python 3.13 (most correct for PyO3 linkage)
    for candidate in python3.13 python3; do
        # Check ALL matching binaries on PATH, not just the first
        while IFS= read -r py_path; do
            if [[ "$py_path" == /nix/store/* ]] && "$py_path" --version 2>&1 | grep -q "3\.13"; then
                echo "$py_path"
                return 0
            fi
        done < <(type -aP "$candidate" 2>/dev/null)
    done
    # Fallback: any Python 3.13 on PATH (e.g. homebrew)
    for candidate in python3.13 python3; do
        local py_path
        py_path=$(command -v "$candidate" 2>/dev/null) || continue
        if "$py_path" --version 2>&1 | grep -q "3\.13"; then
            echo "$py_path"
            return 0
        fi
    done
    return 1
}

check_nix_shell() {
    # NIX_SHELLRC is set by the elodin flake's shellHook
    if [ -n "${NIX_SHELLRC:-}" ]; then
        return 0
    fi
    # Fallback: check IN_NIX_SHELL (set by some nix versions)
    if [ -n "${IN_NIX_SHELL:-}" ]; then
        return 0
    fi
    # Fallback: check if nix python is on PATH
    if detect_nix_python >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

do_build_bf() {
    step 1 "Check betaflight directory"
    if [ ! -d "$BETAFLIGHT_DIR" ]; then
        fail "Betaflight directory not found at $BETAFLIGHT_DIR"
    fi
    ok "Found $BETAFLIGHT_DIR"

    step 2 "Install ARM SDK (if needed)"
    if [ ! -d "$BETAFLIGHT_DIR/tools" ]; then
        echo "  Downloading ARM toolchain..."
        make -C "$BETAFLIGHT_DIR" arm_sdk_install
    else
        ok "ARM SDK already installed"
    fi

    step 3 "Clean SITL build"
    make -C "$BETAFLIGHT_DIR" TARGET=SITL clean
    ok "Clean complete"

    step 4 "Build betaflight SITL"
    make -C "$BETAFLIGHT_DIR" TARGET=SITL
    ELF="$BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf"
    if [ ! -f "$ELF" ]; then
        fail "Build did not produce $ELF"
    fi
    ok "Built $ELF"

    echo -e "\n${GREEN}══════════════════════════════════════${NC}"
    echo -e "${GREEN}  BETAFLIGHT BUILD COMPLETE${NC}"
    echo -e "${GREEN}══════════════════════════════════════${NC}"
}

do_rebuild() {
    # ── Step 1: Check nix shell ──
    step 1 "Check nix shell"
    if ! check_nix_shell; then
        fail "Not in nix shell. Run 'nix develop' first."
    fi
    ok "In nix shell"

    # ── Step 2: Detect nix Python 3.13 ──
    step 2 "Detect nix Python 3.13"
    NIX_PYTHON=$(detect_nix_python) || fail "Cannot find Python 3.13 from nix store on PATH. Is nix develop active?"
    NIX_PY_VERSION=$("$NIX_PYTHON" --version 2>&1)
    ok "$NIX_PY_VERSION at $NIX_PYTHON"

    # ── Step 3: Create/verify venv with Python 3.13 ──
    step 3 "Setup .venv with Python 3.13"
    NEED_VENV=false
    if [ ! -f .venv/bin/python3 ]; then
        NEED_VENV=true
        warn "No .venv found"
    else
        VENV_PY_VERSION=$(.venv/bin/python3 --version 2>&1)
        if [[ "$VENV_PY_VERSION" != *"3.13"* ]]; then
            NEED_VENV=true
            warn ".venv has $VENV_PY_VERSION, need 3.13"
        else
            ok ".venv already has $VENV_PY_VERSION"
        fi
    fi
    if [ "$NEED_VENV" = true ]; then
        echo "  Creating .venv with Python 3.13..."
        uv venv .venv --python "$NIX_PYTHON" --quiet || fail "uv venv failed"
        ok ".venv created"
    fi

    # ── Step 4: Activate venv + clean environment ──
    step 4 "Activate venv and clean environment"
    source .venv/bin/activate
    unset PYTHONPATH 2>/dev/null || true
    export PYO3_PYTHON="$NIX_PYTHON"
    ACTIVE_PY=$(python3 --version 2>&1)
    ACTIVE_PY_PATH=$(which python3)
    ok "Active: $ACTIVE_PY at $ACTIVE_PY_PATH"
    ok "PYO3_PYTHON=$PYO3_PYTHON"

    # ── Step 5: Build nox-py wheel (includes elodin-db library) ──
    step 5 "Build nox-py wheel (includes elodin-db fix)"
    echo "  This compiles: nox-py + elodin-db + impeller2 + all deps..."
    echo "  (This is the Python SDK that runs the simulation subprocess)"
    maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
    if [ $? -ne 0 ]; then
        fail "maturin develop failed"
    fi
    ok "nox-py wheel built and installed in .venv"

    # ── Step 6: Verify elodin module loads ──
    step 6 "Verify elodin Python module loads"
    python3 -c "import elodin; print(f'  elodin module OK: {elodin.__file__}')" || fail "Cannot import elodin"
    ok "Python module works"

    # ── Step 7: Install editor binary ──
    step 7 "Install editor binary (cargo install)"
    echo "  Installing elodin editor to ~/.cargo/bin..."
    cargo install --path apps/elodin --locked
    if [ $? -ne 0 ]; then
        fail "cargo install elodin failed"
    fi
    ok "Editor binary installed to $(command -v elodin 2>/dev/null || echo '~/.cargo/bin/elodin')"

    echo -e "\n${GREEN}══════════════════════════════════════${NC}"
    echo -e "${GREEN}  BUILD COMPLETE${NC}"
    echo -e "${GREEN}══════════════════════════════════════${NC}"
}

do_run() {
    # Verify elodin binary exists
    ELODIN_BIN="${HOME}/.cargo/bin/elodin"
    if [ ! -x "$ELODIN_BIN" ]; then
        # Try PATH
        ELODIN_BIN=$(command -v elodin 2>/dev/null) || fail "elodin binary not found. Run './run.sh rebuild-elodin' first."
    fi

    # Activate venv if not already active
    if [[ "${VIRTUAL_ENV:-}" != *".venv"* ]]; then
        source .venv/bin/activate
    fi
    unset PYTHONPATH 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$PATH"

    # Pre-flight: verify elodin Python module is importable
    if ! python3 -c "import elodin" 2>/dev/null; then
        fail "elodin Python module not installed in .venv. Run './run.sh rebuild-elodin' (inside nix develop) first."
    fi

    echo -e "\n${GREEN}▶ Running editor with $EXAMPLE${NC}"
    echo -e "  Binary: $ELODIN_BIN"
    echo -e "  Log file: $LOG_FILE"
    echo -e "  Wait ~2 min for warmup, then use joystick."
    echo -e "  After testing, Ctrl+C or Cmd+Q to stop.\n"

    PYTHONUNBUFFERED=1 \
    RUST_LOG=elodin_editor=warn,impeller2_bevy=warn \
        "$ELODIN_BIN" editor "$EXAMPLE" 2>&1 | tee "$LOG_FILE"
}

# ── Test result codes returned by run_single_e2e ──
#   0 = PASS
#   1 = CONTROLLER_FAIL (BF behaved wrongly, test ran)
#   2 = INFRA_FAIL (BF/SITL didn't start cleanly or crashed; not the controller's fault)
RC_PASS=0
RC_CONTROLLER_FAIL=1
RC_INFRA_FAIL=2

# Set by run_single_e2e_inner to the human-readable infra-fail reason whenever
# it returns RC_INFRA_FAIL. Cleared at the start of each call. The retry
# wrapper (run_single_e2e) saves the first-attempt reason to
# LAST_INFRA_REASON_FIRST before retrying, so callers can see both.
LAST_INFRA_REASON=""
LAST_INFRA_REASON_FIRST=""

# Set by run_single_e2e_inner to the BF SITL PID it started. Used by the
# retry wrapper as primary cleanup target (fallback: pattern kill).
LAST_BF_PID=""

# ── Suite stability thresholds ──
# First-attempt infra rate above this in a multi-run suite triggers
# "⚠ SUITE UNSTABLE" header. Tunable as suite execution stabilises.
SUITE_INFRA_THRESHOLD_PCT=5
# Single test with first-attempt infra count >= this in same suite triggers
# "⚠ FLAKY" tag — investigate as a deterministic test-specific issue, not
# random startup race.
TEST_FLAKY_THRESHOLD=2

# ── Wait for BF SITL ports to be free ──
# Returns 0 if all ports free within timeout, 1 otherwise.
# On timeout, prints which ports are still held and by which PIDs / TIME_WAIT.
# Args: $1 = timeout in seconds (default 30 — must accommodate macOS TCP
#       TIME_WAIT, which is 2*MSL = 30s default. Previous 10s was too short
#       and caused repeated `bind port 5761 for UART1 failed!!` infra fails
#       between back-to-back BF launches.)
#
# Why both lsof and netstat: lsof shows only sockets owned by a live process.
# After SIGKILL of BF, the kernel still holds the TCP socket in TIME_WAIT for
# ~30s with no owning PID, so lsof misses it but bind() still fails. We probe
# netstat for TIME_WAIT entries on the TCP-only ports (5761).
wait_ports_free() {
    local timeout_s="${1:-30}"
    local ports=(5761 9001 9002 9003 9004 2240)
    local tcp_ports=(5761)   # ports we additionally probe for TIME_WAIT
    local start; start=$(date +%s)
    while true; do
        local held=()
        local p
        for p in "${ports[@]}"; do
            local pids
            pids=$(lsof -ti :$p 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            [ -n "$pids" ] && held+=("$p(pids=$pids)")
        done
        # Detect kernel-held TIME_WAIT (no PID) on TCP ports.
        # Local Address is netstat field 4. Port separator differs by OS:
        #   macOS:  127.0.0.1.5761  (dot)
        #   Linux:  127.0.0.1:5761  (colon)
        # Anchor on either separator + end-of-field so we don't false-match
        # ephemeral foreign ports or substrings of larger numbers.
        for p in "${tcp_ports[@]}"; do
            if netstat -an -p tcp 2>/dev/null | awk -v port="$p" '
                /TIME_WAIT/ && $4 ~ "[.:]"port"$" {found=1}
                END {exit !found}
            '; then
                held+=("$p(TIME_WAIT)")
            fi
        done
        [ ${#held[@]} -eq 0 ] && return 0
        local now; now=$(date +%s)
        if [ $((now - start)) -ge $timeout_s ]; then
            echo -e "  ${RED}✗ Ports still held after ${timeout_s}s: ${held[*]}${NC}"
            return 1
        fi
        sleep 0.5
    done
}

# ── Post-launch infra-fail detection (polling) ──
# Used after BF launch + boot grace, before the Python test starts. Polls for
# up to N seconds because BF may take a moment to start writing logs.
# Args: $1=log file, $2=BF PID, $3=poll timeout in seconds (default 5)
# Echoes a reason string if infra-fail; empty if BF appears healthy and ready.
detect_infra_fail_startup() {
    local log="$1" bf_pid="$2" poll_timeout="${3:-5}"
    local start; start=$(date +%s)
    while true; do
        local alive=0
        kill -0 "$bf_pid" 2>/dev/null && alive=1

        # Fatal signature in log (regardless of alive/dead)
        if [ -s "$log" ]; then
            local matched
            matched=$(grep -m1 -E 'bind port .* failed|Segmentation fault|Trace/BPT trap|Bus error' "$log" 2>/dev/null)
            if [ -n "$matched" ]; then
                echo "fatal-init-signature: $matched"
                return 0
            fi
        fi

        # SITL dead
        if [ $alive -eq 0 ]; then
            if [ ! -s "$log" ]; then echo "sitl-died-no-output"
            else echo "sitl-exited-during-startup"
            fi
            return 0
        fi

        # SITL alive + log has content → ready, no infra-fail
        if [ -s "$log" ]; then
            echo ""
            return 0
        fi

        # SITL alive + zero-byte log: poll until timeout
        local now; now=$(date +%s)
        if [ $((now - start)) -ge $poll_timeout ]; then
            echo "startup-timeout"
            return 0
        fi
        sleep 0.2
    done
}

# ── Post-test infra-fail detection (single check) ──
# Used after the Python test exits non-zero, to distinguish controller-fail
# from "SITL crashed mid-test". Conservative: only flags hard infra signs
# (fatal signature OR dead PID). "Alive + silent + no fatal" → controller fail.
# Args: $1=log file, $2=BF PID
# Echoes reason if infra-fail; empty if test exit should be attributed to controller.
detect_infra_fail_posttest() {
    local log="$1" bf_pid="$2"
    if [ -s "$log" ]; then
        local matched
        matched=$(grep -m1 -E 'bind port .* failed|Segmentation fault|Trace/BPT trap|Bus error' "$log" 2>/dev/null)
        if [ -n "$matched" ]; then
            echo "fatal-init-signature: $matched"
            return 0
        fi
        # SITL alive but unresponsive: BF process is up but never released the
        # lockstep mutex (or UDP receiver is dead). Test sees sustained motor
        # response timeouts; comms.py raises TimeoutError after the configured
        # threshold; Python exits non-zero. Without this check the runner would
        # mis-classify as ctrl-fail, since SITL is alive and no fatal signature.
        if grep -qE 'TimeoutError|never responded after|consecutive timeouts' "$log" 2>/dev/null; then
            echo "sitl-alive-but-unresponsive (TimeoutError raised)"
            return 0
        fi
        # Bulk-timeout heuristic: many "Motor timeout" / "Motor response timeout"
        # warnings indicate the same condition even if comms.py didn't trip its
        # threshold yet. Different test scripts emit different warning strings:
        #   - failsafe-althold etc.: "WARNING: Motor response timeout" (3 words)
        #   - smooth-takeoff etc.:   "WARNING: Motor timeout"          (2 words)
        # The optional "( response)?" group catches both. 50 chosen as a safe
        # floor — normal runs have 0; a few timeouts can happen during BF task
        # scheduling jitter; 50+ is unambiguously broken.
        local timeout_count
        timeout_count=$(grep -cE 'Motor( response)? timeout' "$log" 2>/dev/null | tr -d ' \n')
        if [ -n "$timeout_count" ] && [ "$timeout_count" -gt 50 ] 2>/dev/null; then
            echo "sitl-unresponsive (${timeout_count} motor timeouts)"
            return 0
        fi
    fi
    if ! kill -0 "$bf_pid" 2>/dev/null; then
        if [ ! -s "$log" ]; then echo "sitl-dead-no-output"
        else echo "sitl-died-during-test"
        fi
        return 0
    fi
    echo ""
}

# ── Headless E2E runner (single test, single attempt — no retry) ──
# Use run_single_e2e (the wrapper) instead unless you specifically need to
# bypass retry. Returns one of:
#   $RC_PASS (0)             — test passed
#   $RC_CONTROLLER_FAIL (1)  — test ran, BF/test reported failure
#   $RC_INFRA_FAIL (2)       — BF SITL didn't start cleanly or crashed
# Sets LAST_INFRA_REASON on infra-fail. Sets LAST_BF_PID to the BF process
# it spawned (so wrapper can target-kill before retry).
run_single_e2e_inner() {
    local test_script="$1"
    local test_name="$2"
    local log_file="/tmp/bf-e2e-${test_name}.log"
    LAST_INFRA_REASON=""   # cleared per call; set before any RC_INFRA_FAIL return
    LAST_BF_PID=""         # cleared per call; set after SITL spawn

    # Verify test script exists
    if [ ! -f "$test_script" ]; then
        echo -e "  ${RED}✗ Test script not found: $test_script${NC}"
        return 1
    fi

    # Activate venv if not already active
    if [[ "${VIRTUAL_ENV:-}" != *".venv"* ]]; then
        source .venv/bin/activate
    fi
    unset PYTHONPATH 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$PATH"

    # Pre-flight: verify elodin Python module
    if ! python3 -c "import elodin" 2>/dev/null; then
        echo -e "  ${RED}✗ elodin Python module not installed. Run './run.sh rebuild-elodin' first.${NC}"
        return 1
    fi

    # Verify BF SITL binary
    local elf="$BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf"
    if [ ! -f "$elf" ]; then
        echo -e "  ${RED}✗ Betaflight SITL not found at $elf. Run './run.sh build-bf' first.${NC}"
        return 1
    fi

    # Kill stale processes from previous test
    pkill -f betaflight_SITL 2>/dev/null || true
    # Kill any lingering elodin-db / Python test processes holding port 2240,
    # but ONLY if the holder is recognizably ours. Avoid SIGKILL-ing unrelated
    # tools that happen to also use port 2240 (elodin-db's default port).
    local pid
    for pid in $(lsof -ti :2240 2>/dev/null); do
        local comm
        comm=$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ')
        case "$comm" in
            *elodin*|*python*|*betaflight*)
                kill -9 "$pid" 2>/dev/null || true
                ;;
            "")
                # Process already gone — nothing to do
                ;;
            *)
                echo -e "  ${YELLOW}⚠ Port 2240 held by unrelated process pid=$pid comm=$comm — NOT killing${NC}"
                ;;
        esac
    done
    # Force kill any remaining BF
    pkill -9 -f betaflight_SITL 2>/dev/null || true
    # Clean up ALL stale elodin test DB directories
    rm -rf e2e_*_db e2e_*_test_db /tmp/e2e_*_db 2>/dev/null || true

    # Active port-readiness wait (replaces bare sleep). If ports are still
    # held after timeout, classify as INFRA_FAIL — port collision is an
    # infrastructure issue, not a controller bug.
    if ! wait_ports_free 30; then
        LAST_INFRA_REASON="ports-still-held"
        echo -e "  ${RED}✗ INFRA_FAIL: ports-still-held${NC}"
        return $RC_INFRA_FAIL
    fi

    echo -e "\n${GREEN}▶ Running E2E test (headless): $test_name${NC}"
    echo -e "  Script: $test_script"
    echo -e "  Betaflight: $elf"
    echo -e "  Log file: $log_file"

    # Start Betaflight SITL in background
    local bf_pid=""
    (cd "$BETAFLIGHT_DIR" && exec "$elf") > "$log_file" 2>&1 &
    bf_pid=$!
    LAST_BF_PID="$bf_pid"   # exposed for retry wrapper's targeted cleanup

    # Trap-based cleanup: guarantee SITL is killed on any exit from this function
    _e2e_cleanup() {
        if [ -n "$bf_pid" ]; then
            kill "$bf_pid" 2>/dev/null || true
            wait "$bf_pid" 2>/dev/null || true
            bf_pid=""
        fi
    }
    trap _e2e_cleanup RETURN

    echo -e "  Betaflight SITL started (PID $bf_pid) from $BETAFLIGHT_DIR"
    sleep 2  # boot grace — wait for UDP port binding before infra check

    # Startup infra-fail detection (polls up to 5s for log content / fatal sig / death)
    local startup_reason
    startup_reason=$(detect_infra_fail_startup "$log_file" "$bf_pid" 5)
    if [ -n "$startup_reason" ]; then
        LAST_INFRA_REASON="$startup_reason"
        echo -e "  ${RED}✗ INFRA_FAIL during startup: ${startup_reason}${NC}"
        trap - RETURN
        _e2e_cleanup
        return $RC_INFRA_FAIL
    fi

    # Run test — capture exit code without triggering set -e
    local test_exit=0
    PYTHONUNBUFFERED=1 \
        python3 "$test_script" run --no-s10 2>&1 | tee -a "$log_file" || test_exit=$?

    # Post-test infra-fail detection: if test failed, scan the log for fatal
    # signatures + check PID liveness before attributing to the controller.
    # SITL crashes mid-test would otherwise look like controller failures.
    if [ $test_exit -ne 0 ]; then
        local posttest_reason
        posttest_reason=$(detect_infra_fail_posttest "$log_file" "$bf_pid")
        if [ -n "$posttest_reason" ]; then
            LAST_INFRA_REASON="$posttest_reason"
            echo -e "  ${YELLOW}⚠ INFRA_FAIL during test: ${posttest_reason}${NC}"
            trap - RETURN
            _e2e_cleanup
            return $RC_INFRA_FAIL
        fi
    fi

    # Clear trap before normal cleanup
    trap - RETURN
    _e2e_cleanup

    # test_exit is 0 (PASS) or non-zero (CONTROLLER_FAIL).
    # Map non-zero to RC_CONTROLLER_FAIL for clarity at the call site.
    [ $test_exit -eq 0 ] && return $RC_PASS || return $RC_CONTROLLER_FAIL
}

# ── Headless E2E runner (single test, with infra-fail retry) ──
# Wrapper around run_single_e2e_inner that retries ONCE on specific startup-
# race signatures (sitl-died-no-output, sitl-exited-during-startup,
# startup-timeout). Other infra reasons (ports-still-held, fatal-init-
# signature, mid-test crashes) are NOT retried — they're either deterministic
# or indicate deeper issues worth investigating.
#
# Controller-fails are NEVER retried.
#
# Set E2E_NO_RETRY=1 to disable retry (useful when investigating).
#
# Sets LAST_INFRA_REASON_FIRST to the first-attempt reason (always).
# Sets LAST_INFRA_REASON to the second-attempt reason if retry was triggered
# and still infra-failed (otherwise the value from the successful attempt or
# the first attempt's reason if no retry was triggered).
#
# Retry uses a separate log file: /tmp/bf-e2e-<test_name>-attempt2.log
# (attempt 1 log preserved for inspection).
run_single_e2e() {
    local test_script="$1"
    local test_name="$2"
    LAST_INFRA_REASON_FIRST=""

    local rc=0
    run_single_e2e_inner "$test_script" "$test_name" || rc=$?

    if [ $rc -ne $RC_INFRA_FAIL ]; then
        return $rc   # PASS or CONTROLLER_FAIL — never retry
    fi

    LAST_INFRA_REASON_FIRST="$LAST_INFRA_REASON"
    local first_reason="$LAST_INFRA_REASON"

    # Only retry on startup-race / sitl-state signatures.
    # sitl-alive-but-unresponsive and sitl-unresponsive are SITL state issues
    # (lockstep mutex, UDP receiver) that retry-with-fresh-process can recover.
    # The fatal-init-signature: bind port branch is narrowly retry-eligible
    # (kernel-held TIME_WAIT slipping past wait_ports_free); other
    # fatal-init-signature reasons (Segmentation fault, Bus error, Trace/BPT
    # trap) MUST surface as hard fails, not loop.
    case "$first_reason" in
        sitl-died-no-output*|sitl-exited-during-startup*|startup-timeout*|sitl-alive-but-unresponsive*|sitl-unresponsive*|"fatal-init-signature: bind port"*)
            ;;
        *)
            echo -e "  ${YELLOW}⚠ INFRA (${first_reason}) — not retry-eligible${NC}"
            return $RC_INFRA_FAIL
            ;;
    esac

    # Escape hatch for diagnostics
    if [ "${E2E_NO_RETRY:-0}" = "1" ]; then
        echo -e "  ${YELLOW}⚠ INFRA (${first_reason}); retry disabled by E2E_NO_RETRY${NC}"
        return $RC_INFRA_FAIL
    fi

    echo -e "  ${YELLOW}⚠ INFRA on attempt 1 (${first_reason}); aggressive cleanup + attempt 2${NC}"

    # Targeted PID kill first (idempotent — trap-cleanup already ran in inner),
    # then pattern kill as fallback for any leaked processes.
    if [ -n "$LAST_BF_PID" ]; then
        kill -9 "$LAST_BF_PID" 2>/dev/null || true
    fi
    pkill -9 -f betaflight_SITL 2>/dev/null || true
    sleep 5

    # Attempt 2 — separate log file so attempt 1 is preserved for inspection
    local rc2=0
    run_single_e2e_inner "$test_script" "${test_name}-attempt2" || rc2=$?
    return $rc2
}

# ── Focused suite for IGE-sensitive physics regression ──
# Reproduces real-hardware "floating"/"moon takeoff" behavior under realistic
# IGE. Used by `e2e-focused` target (defaults realistic, 5 cycles).
# low-alt-horizontal added in v10.4 — tests spin-lock predicate during low-alt
# flight, the scenario that caught v10.3.6→v10.3.7 sticky-snapshot regression.
FOCUSED_TESTS=(
    "ground-idle:$E2E_GROUND_IDLE_SCRIPT"
    "nosettle-takeoff:$E2E_NOSETTLE_SCRIPT"
    "failsafe-althold:$E2E_FAILSAFE_SCRIPT"
    "failsafe-init:$E2E_FAILSAFE_INIT_SCRIPT"
    "midair-activation:$E2E_MIDAIR_SCRIPT"
    "low-alt-horizontal:$E2E_LOW_ALT_HORIZONTAL_SCRIPT"
)

# ── Wind suite (ALTHOLD) ──
# Tests that exercise ALTHOLD behavior under XY wind disturbance — covers
# both ground-arm scenarios (ground-idle: drone armed in wind) and flight
# scenarios (angle/acro/failsafe/midair/low-alt). Used by `e2e-wind-althold`
# target; defaults realistic physics + moderate wind, both overridable via
# E2E_PHYSICS_PROFILE and E2E_WIND_PROFILE env vars.
# v10.4: added ground-idle (real-hardware "arm in wind" matters) and
# low-alt-horizontal (spin-lock predicate stress under wind perturbations).
WIND_TESTS=(
    "ground-idle:$E2E_GROUND_IDLE_SCRIPT"
    "angle-althold:$E2E_SCRIPT"
    "acro-althold:$E2E_ACRO_SCRIPT"
    "failsafe-althold:$E2E_FAILSAFE_SCRIPT"
    "midair-activation:$E2E_MIDAIR_SCRIPT"
    "low-alt-horizontal:$E2E_LOW_ALT_HORIZONTAL_SCRIPT"
)

# ── POSHOLD wind suite ──
# POSHOLD stress-tests XY drift compensation under wind. Separate from
# ALTHOLD suite because POSHOLD failure modes (oscillation, integral wind-up)
# differ from altitude-hold ones.
WIND_POSHOLD_TESTS=(
    "poshold:$E2E_POSHOLD_SCRIPT"
)

# Parse a runs argument. Accepts "5", "--runs=5", or empty (defaults to 1).
parse_runs() {
    local arg="${1:-1}"
    case "$arg" in
        --runs=*) echo "${arg#--runs=}" ;;
        ''|*[!0-9]*) echo "1" ;;
        *) echo "$arg" ;;
    esac
}

# Run a SINGLE test multiple times with per-run logs.
# Mirrors do_focused_suite's accounting (pass / ctrl-fail / infra-first /
# infra-final / recovered) but for one test only.
#
# Log naming:
#   runs=1  → /tmp/bf-e2e-${test_name}.log                (single-run, legacy)
#   runs>1  → /tmp/bf-e2e-[${profile}-]${test_name}-rN.log
# The profile prefix matches do_focused_suite so logs from manual single-test
# multi-runs and focused-suite multi-runs use the same convention.
#
# Args: $1=test_script, $2=test_name, $3=runs arg (parse_runs format)
# Return: 0 = all pass, 1 = any controller fail, 2 = infra-only failures
do_single_test_runs() {
    local test_script="$1"
    local test_name="$2"
    local runs
    runs=$(parse_runs "${3:-1}")

    # Default path: legacy single-run, preserves existing log filename.
    if [ "$runs" -le 1 ]; then
        run_single_e2e "$test_script" "$test_name"
        return $?
    fi

    local profile_prefix=""
    if [ -n "${E2E_PHYSICS_PROFILE:-}" ]; then
        profile_prefix="${E2E_PHYSICS_PROFILE}-"
    fi

    local passed=0 ctrl_fail=0 infra_first=0 infra_final=0 recovered=0
    local infra_reasons=()
    local suite_start
    suite_start=$(date +%s)

    echo ""
    echo "========================================================================"
    echo "  SINGLE TEST MULTI-RUN — ${test_name}  runs=${runs}  profile=${E2E_PHYSICS_PROFILE:-baseline}"
    echo "========================================================================"

    local i=0
    while [ "$i" -lt "$runs" ]; do
        i=$((i + 1))
        echo -e "\n────── Run $i / $runs ──────"

        local rc=0
        run_single_e2e "$test_script" "${profile_prefix}${test_name}-r${i}" || rc=$?

        local was_retried=0
        if [ -n "$LAST_INFRA_REASON_FIRST" ]; then
            case "$LAST_INFRA_REASON_FIRST" in
                sitl-died-no-output*|sitl-exited-during-startup*|startup-timeout*|sitl-alive-but-unresponsive*|sitl-unresponsive*|"fatal-init-signature: bind port"*) was_retried=1 ;;
            esac
            if [ "${E2E_NO_RETRY:-0}" = "1" ]; then was_retried=0; fi
            infra_first=$((infra_first + 1))
        fi

        case $rc in
            0)
                passed=$((passed + 1))
                if [ $was_retried -eq 1 ]; then
                    recovered=$((recovered + 1))
                    infra_reasons+=("r${i}: ${LAST_INFRA_REASON_FIRST} → [recovered after retry]")
                fi
                ;;
            2)
                infra_final=$((infra_final + 1))
                if [ $was_retried -eq 1 ]; then
                    infra_reasons+=("r${i}: ${LAST_INFRA_REASON_FIRST} → STILL INFRA after retry: ${LAST_INFRA_REASON}")
                else
                    # No retry happened: ensure first-attempt infra is counted
                    if [ -z "$LAST_INFRA_REASON_FIRST" ]; then
                        infra_first=$((infra_first + 1))
                    fi
                    infra_reasons+=("r${i}: ${LAST_INFRA_REASON:-unknown} [not retry-eligible]")
                fi
                ;;
            *)
                ctrl_fail=$((ctrl_fail + 1))
                if [ $was_retried -eq 1 ]; then
                    recovered=$((recovered + 1))
                    infra_reasons+=("r${i}: ${LAST_INFRA_REASON_FIRST} → ctrl-fail after retry")
                fi
                ;;
        esac
    done

    local suite_end
    suite_end=$(date +%s)
    local suite_duration=$((suite_end - suite_start))
    local suite_min=$((suite_duration / 60))
    local suite_sec=$((suite_duration % 60))

    local effective_total=$((passed + ctrl_fail))
    local first_infra_pct=0
    if [ $runs -gt 0 ]; then
        first_infra_pct=$(( (infra_first * 100) / runs ))
    fi

    echo ""
    echo "========================================================================"
    echo "  SINGLE TEST RESULTS — ${test_name}  runs=${runs}  profile=${E2E_PHYSICS_PROFILE:-baseline}"
    echo "========================================================================"
    echo "  Total: $runs  Pass: $passed  Ctrl-fail: $ctrl_fail  Infra-fail (final): $infra_final  Duration: ${suite_min}m ${suite_sec}s"
    echo "  First-attempt infra: $infra_first (${first_infra_pct}%) — $recovered recovered by retry, $infra_final still infra after retry"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $passed/$effective_total (excludes final infra failures)"
    else
        echo "  Effective pass rate: n/a (all runs were infra-fails)"
    fi

    if [ $first_infra_pct -gt $SUITE_INFRA_THRESHOLD_PCT ]; then
        echo ""
        echo -e "  ${YELLOW}⚠ UNSTABLE — first-attempt infra rate ${first_infra_pct}% > ${SUITE_INFRA_THRESHOLD_PCT}% threshold${NC}"
    fi

    if [ ${#infra_reasons[@]} -gt 0 ]; then
        echo ""
        echo "  Infra-fail reasons (first attempt → retry outcome):"
        local r
        for r in "${infra_reasons[@]}"; do
            echo "    $r"
        done
    fi

    echo ""
    echo "  Per-run logs:"
    echo "    Attempt 1: /tmp/bf-e2e-${profile_prefix}${test_name}-r<N>.log"
    echo "    Attempt 2: /tmp/bf-e2e-${profile_prefix}${test_name}-r<N>-attempt2.log (only if retried)"
    echo "========================================================================"

    if   [ $ctrl_fail   -gt 0 ]; then return 1
    elif [ $infra_final -gt 0 ]; then return 2
    else                              return 0
    fi
}

# Run focused suite under a given E2E_PHYSICS_PROFILE for N runs.
# Tracks per-test pass / controller-fail / infra-fail counts and prints a
# three-column summary. Effective pass rate excludes infra-fails so the
# number reflects controller behavior, not infrastructure noise.
#
# Return codes:
#   0 — all runs passed
#   1 — at least one controller failure
#   2 — infra-only failures (no controller failures)
do_focused_suite() {
    local profile="$1"
    local runs
    runs=$(parse_runs "${2:-1}")

    export E2E_PHYSICS_PROFILE="$profile"

    local total_tests=${#FOCUSED_TESTS[@]}
    local total_runs=$((runs * total_tests))
    local total_passed=0
    local total_ctrl_fail=0
    local total_infra_final=0       # final classification (after retry)
    local total_infra_first=0       # first-attempt infra (always counted, transparent)
    local total_recovered=0         # retried and succeeded (no longer infra)

    # Per-test counters. Counters bucket by BASE test name (not -attempt2),
    # so retry preserves test identity in aggregation.
    local test_names=()
    local test_passes=()
    local test_ctrl_fails=()
    local test_infra_first=()       # first-attempt infra per test
    local test_infra_final=()       # final infra (after retry) per test
    for entry in "${FOCUSED_TESTS[@]}"; do
        test_names+=("${entry%%:*}")
        test_passes+=(0)
        test_ctrl_fails+=(0)
        test_infra_first+=(0)
        test_infra_final+=(0)
    done

    # Per-incident records. Records BOTH first and second reasons when retry
    # was triggered, so reader can distinguish transient race vs deterministic
    # crash (where same signature reappears) vs different failure modes.
    local infra_reasons=()

    local suite_start
    suite_start=$(date +%s)

    echo ""
    echo "========================================================================"
    echo "  FOCUSED SUITE — profile=$profile  runs=$runs  tests=$total_tests"
    echo "========================================================================"

    local run_idx=0
    while [ "$run_idx" -lt "$runs" ]; do
        run_idx=$((run_idx + 1))
        echo -e "\n────── Run $run_idx / $runs ──────"

        local i=0
        for entry in "${FOCUSED_TESTS[@]}"; do
            local name="${entry%%:*}"
            local script="${entry#*:}"

            echo -e "\n━━━ [run $run_idx] $name ━━━"

            local rc=0
            run_single_e2e "$script" "${profile}-${name}-r${run_idx}" || rc=$?

            # Did wrapper trigger a retry? Yes if LAST_INFRA_REASON_FIRST set.
            local was_retried=0
            if [ -n "$LAST_INFRA_REASON_FIRST" ]; then
                was_retried=1
                # Also: even non-retry-eligible infras populate FIRST. To avoid
                # false retry-counted, only count as retried if reason was eligible.
                case "$LAST_INFRA_REASON_FIRST" in
                    sitl-died-no-output*|sitl-exited-during-startup*|startup-timeout*|sitl-alive-but-unresponsive*|sitl-unresponsive*|"fatal-init-signature: bind port"*) ;;
                    *) was_retried=0 ;;
                esac
                if [ "${E2E_NO_RETRY:-0}" = "1" ]; then
                    was_retried=0
                fi
                # Always count first-attempt infra
                total_infra_first=$((total_infra_first + 1))
                test_infra_first[$i]=$((${test_infra_first[$i]} + 1))
            fi

            case $rc in
                0)  # RC_PASS — possibly recovered by retry
                    test_passes[$i]=$((${test_passes[$i]} + 1))
                    total_passed=$((total_passed + 1))
                    if [ $was_retried -eq 1 ]; then
                        total_recovered=$((total_recovered + 1))
                        infra_reasons+=("${name} r${run_idx}: ${LAST_INFRA_REASON_FIRST} → [recovered after retry]")
                    fi
                    ;;
                2)  # RC_INFRA_FAIL — final
                    test_infra_final[$i]=$((${test_infra_final[$i]} + 1))
                    total_infra_final=$((total_infra_final + 1))
                    if [ $was_retried -eq 1 ]; then
                        infra_reasons+=("${name} r${run_idx}: ${LAST_INFRA_REASON_FIRST} → STILL INFRA after retry: ${LAST_INFRA_REASON}")
                    else
                        # First-attempt infra not yet counted (no retry triggered)
                        if [ -z "$LAST_INFRA_REASON_FIRST" ]; then
                            total_infra_first=$((total_infra_first + 1))
                            test_infra_first[$i]=$((${test_infra_first[$i]} + 1))
                        fi
                        infra_reasons+=("${name} r${run_idx}: ${LAST_INFRA_REASON:-unknown} [not retry-eligible]")
                    fi
                    ;;
                *)  # RC_CONTROLLER_FAIL or any other non-zero
                    test_ctrl_fails[$i]=$((${test_ctrl_fails[$i]} + 1))
                    total_ctrl_fail=$((total_ctrl_fail + 1))
                    if [ $was_retried -eq 1 ]; then
                        # Recovered to controller-fail (rare but possible)
                        total_recovered=$((total_recovered + 1))
                        infra_reasons+=("${name} r${run_idx}: ${LAST_INFRA_REASON_FIRST} → ctrl-fail after retry")
                    fi
                    ;;
            esac
            i=$((i + 1))
        done
    done

    local suite_end
    suite_end=$(date +%s)
    local suite_duration=$((suite_end - suite_start))
    local suite_min=$((suite_duration / 60))
    local suite_sec=$((suite_duration % 60))

    # Effective pass rate excludes FINAL infra failures (the controller-meaningful denominator)
    local effective_total=$((total_passed + total_ctrl_fail))

    # Stability assessment: first-attempt infra rate
    local first_infra_pct=0
    if [ $total_runs -gt 0 ]; then
        first_infra_pct=$(( (total_infra_first * 100) / total_runs ))
    fi

    echo ""
    echo "========================================================================"
    echo "  FOCUSED SUITE RESULTS — profile=$profile  runs=$runs"
    echo "========================================================================"
    echo "  Total runs: $total_runs  Pass: $total_passed  Ctrl-fail: $total_ctrl_fail  Infra-fail (final): $total_infra_final  Duration: ${suite_min}m ${suite_sec}s"
    echo "  First-attempt infra: $total_infra_first (${first_infra_pct}%) — $total_recovered recovered by retry, $total_infra_final still infra after retry"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $total_passed/$effective_total (excludes final infra failures)"
    else
        echo "  Effective pass rate: n/a (all runs were infra-fails)"
    fi

    if [ $first_infra_pct -gt $SUITE_INFRA_THRESHOLD_PCT ]; then
        echo ""
        echo -e "  ${YELLOW}⚠ SUITE UNSTABLE — first-attempt infra rate ${first_infra_pct}% > ${SUITE_INFRA_THRESHOLD_PCT}% threshold${NC}"
    fi

    echo ""
    echo "  Per-test breakdown (across $runs runs):"
    printf "    %-22s  %-7s  %-9s  %-11s  %-13s  %s\n" "test" "pass" "ctrl-fail" "infra (1st)" "infra (final)" "flag"
    local i=0
    for name in "${test_names[@]}"; do
        local p="${test_passes[$i]}"
        local cf="${test_ctrl_fails[$i]}"
        local infra1="${test_infra_first[$i]}"
        local infraF="${test_infra_final[$i]}"
        local flag=""
        if [ $infra1 -ge $TEST_FLAKY_THRESHOLD ]; then
            flag="⚠ FLAKY"
        fi
        printf "    %-22s  %d/%d    %d/%d      %d/%d         %d/%d           %s\n" "$name" "$p" "$runs" "$cf" "$runs" "$infra1" "$runs" "$infraF" "$runs" "$flag"
        i=$((i + 1))
    done

    if [ ${#infra_reasons[@]} -gt 0 ]; then
        echo ""
        echo "  Infra-fail reasons (first attempt → retry outcome):"
        local r
        for r in "${infra_reasons[@]}"; do
            echo "    $r"
        done
    fi

    echo ""
    echo "  Per-run logs:"
    echo "    Attempt 1: /tmp/bf-e2e-${profile}-<test>-r<N>.log"
    echo "    Attempt 2: /tmp/bf-e2e-${profile}-<test>-r<N>-attempt2.log (only if retried)"
    echo "========================================================================"

    # Three-state return: 0 = all pass, 1 = any controller fail, 2 = infra-only failures
    if [ $total_ctrl_fail -gt 0 ]; then return 1
    elif [ $total_infra_final -gt 0 ]; then return 2
    else                                    return 0
    fi
}

# ── Wind investigation suite ──
# Args: $1 = tests-array-name ("WIND_TESTS" or "WIND_POSHOLD_TESTS"),
#       $2 = runs arg (parse_runs format), defaults to 1
#       $3 = suite display label (e.g. "ALTHOLD", "POSHOLD")
#
# Reads E2E_PHYSICS_PROFILE + E2E_WIND_PROFILE from env (validated in
# config.py). v10.4: physics defaults to realistic (set at script top); wind
# defaults to moderate (set here for wind-suite semantics).
#
# Log path: /tmp/bf-e2e-${physics}_${wind}-${test}-r${N}.log
# (combined profile prefix so wind runs don't overwrite physics-only runs)
do_wind_suite() {
    local tests_var="$1"
    local runs
    runs=$(parse_runs "${2:-1}")
    local suite_label="${3:-wind}"

    # Default wind profile to "moderate" for wind-suite if not set by user.
    # Physics profile defaults to "realistic" (set at script top); preserved
    # if user explicitly overrides via env var.
    if [ -z "${E2E_WIND_PROFILE:-}" ]; then
        export E2E_WIND_PROFILE="moderate"
    fi
    local physics="${E2E_PHYSICS_PROFILE}"
    local wind="${E2E_WIND_PROFILE}"
    local log_prefix="${physics}_${wind}"

    # bash 3.2 portable: expand tests array via eval
    eval 'local tests=( "${'"$tests_var"'[@]}" )'

    local total_tests=${#tests[@]}
    local total_runs=$((runs * total_tests))
    local total_passed=0 total_ctrl_fail=0 total_infra_final=0
    local total_infra_first=0 total_recovered=0

    local test_names=()
    local test_passes=()
    local test_ctrl_fails=()
    local test_infra_first=()
    local test_infra_final=()
    for entry in "${tests[@]}"; do
        test_names+=("${entry%%:*}")
        test_passes+=(0)
        test_ctrl_fails+=(0)
        test_infra_first+=(0)
        test_infra_final+=(0)
    done

    local suite_start
    suite_start=$(date +%s)

    echo ""
    echo "========================================================================"
    echo "  WIND SUITE — ${suite_label}  physics=${physics}  wind=${wind}  runs=${runs}  tests=${total_tests}"
    echo "========================================================================"

    local run_idx=0
    while [ "$run_idx" -lt "$runs" ]; do
        run_idx=$((run_idx + 1))
        echo -e "\n────── Run $run_idx / $runs ──────"

        local i=0
        for entry in "${tests[@]}"; do
            local name="${entry%%:*}"
            local script="${entry#*:}"

            echo -e "\n━━━ [run $run_idx] $name ━━━"

            local rc=0
            run_single_e2e "$script" "${log_prefix}-${name}-r${run_idx}" || rc=$?

            local was_retried=0
            if [ -n "$LAST_INFRA_REASON_FIRST" ]; then
                case "$LAST_INFRA_REASON_FIRST" in
                    sitl-died-no-output*|sitl-exited-during-startup*|startup-timeout*|sitl-alive-but-unresponsive*|sitl-unresponsive*|"fatal-init-signature: bind port"*) was_retried=1 ;;
                esac
                if [ "${E2E_NO_RETRY:-0}" = "1" ]; then was_retried=0; fi
                total_infra_first=$((total_infra_first + 1))
                test_infra_first[$i]=$((${test_infra_first[$i]} + 1))
            fi

            case $rc in
                0)
                    test_passes[$i]=$((${test_passes[$i]} + 1))
                    total_passed=$((total_passed + 1))
                    [ $was_retried -eq 1 ] && total_recovered=$((total_recovered + 1))
                    ;;
                2)
                    test_infra_final[$i]=$((${test_infra_final[$i]} + 1))
                    total_infra_final=$((total_infra_final + 1))
                    if [ -z "$LAST_INFRA_REASON_FIRST" ]; then
                        total_infra_first=$((total_infra_first + 1))
                        test_infra_first[$i]=$((${test_infra_first[$i]} + 1))
                    fi
                    ;;
                *)
                    test_ctrl_fails[$i]=$((${test_ctrl_fails[$i]} + 1))
                    total_ctrl_fail=$((total_ctrl_fail + 1))
                    [ $was_retried -eq 1 ] && total_recovered=$((total_recovered + 1))
                    ;;
            esac
            i=$((i + 1))
        done
    done

    local suite_end
    suite_end=$(date +%s)
    local suite_duration=$((suite_end - suite_start))
    local suite_min=$((suite_duration / 60))
    local suite_sec=$((suite_duration % 60))

    local effective_total=$((total_passed + total_ctrl_fail))

    echo ""
    echo "========================================================================"
    echo "  WIND SUITE RESULTS — ${suite_label}  physics=${physics}  wind=${wind}  runs=${runs}"
    echo "========================================================================"
    echo "  Total runs: $total_runs  Pass: $total_passed  Ctrl-fail: $total_ctrl_fail  Infra-fail (final): $total_infra_final  Duration: ${suite_min}m ${suite_sec}s"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $total_passed/$effective_total (excludes final infra failures)"
    fi

    echo ""
    echo "  Per-test breakdown (across $runs runs):"
    printf "    %-22s  %-7s  %-9s  %-13s\n" "test" "pass" "ctrl-fail" "infra (final)"
    local i=0
    for name in "${test_names[@]}"; do
        printf "    %-22s  %d/%d    %d/%d      %d/%d\n" "$name" "${test_passes[$i]}" "$runs" "${test_ctrl_fails[$i]}" "$runs" "${test_infra_final[$i]}" "$runs"
        i=$((i + 1))
    done

    echo ""
    echo "  Per-run logs: /tmp/bf-e2e-${log_prefix}-<test>-r<N>.log"
    echo "========================================================================"

    if   [ $total_ctrl_fail -gt 0 ];  then return 1
    elif [ $total_infra_final -gt 0 ]; then return 2
    else                                    return 0
    fi
}

# ── Run all automated E2E tests sequentially, optionally for N cycles ──
# Args: $1 = runs arg (parse_runs format). Defaults to 1.
#
# Log naming:
#   runs=1 → /tmp/bf-e2e-${test_name}.log                  (legacy, single cycle)
#   runs>1 → /tmp/bf-e2e-${test_name}-r${cycle}.log        (per cycle preserved)
#
# Different tests within a cycle never share log paths (each has a unique
# test_name). Across cycles, the -rN suffix keeps cycle N's logs separate
# from cycle N+1's. Nothing is overwritten.
do_e2e_all() {
    local runs
    runs=$(parse_runs "${1:-1}")

    local tests=(
        "e2e-ground-idle:$E2E_GROUND_IDLE_SCRIPT"
        "e2e-smooth-takeoff:$E2E_SMOOTH_TAKEOFF_SCRIPT"
        "e2e-center-semantics:$E2E_CENTER_SCRIPT"
        "e2e-nosettle-takeoff:$E2E_NOSETTLE_SCRIPT"
        "e2e-angle-althold:$E2E_SCRIPT"
        "e2e-acro-althold:$E2E_ACRO_SCRIPT"
        "e2e-flight:$E2E_FLIGHT_SCRIPT"
        "e2e-failsafe-althold:$E2E_FAILSAFE_SCRIPT"
        "e2e-failsafe-init:$E2E_FAILSAFE_INIT_SCRIPT"
        "e2e-midair-activation:$E2E_MIDAIR_SCRIPT"
        "e2e-poshold:$E2E_POSHOLD_SCRIPT"
        "e2e-low-alt-horizontal:$E2E_LOW_ALT_HORIZONTAL_SCRIPT"
    )

    local num_tests=${#tests[@]}
    local total_runs=$((runs * num_tests))

    # Per-test aggregated counters (across all cycles)
    local test_names=()
    local test_passes=()
    local test_ctrl_fails=()
    local test_infra_first=()
    local test_infra_final=()
    for entry in "${tests[@]}"; do
        test_names+=("${entry%%:*}")
        test_passes+=(0)
        test_ctrl_fails+=(0)
        test_infra_first+=(0)
        test_infra_final+=(0)
    done

    local total_passed=0
    local total_ctrl_fail=0
    local total_infra_final=0
    local total_infra_first=0
    local total_recovered=0
    local results=()

    local suite_start
    suite_start=$(date +%s)

    echo ""
    echo "========================================================================"
    if [ "$runs" -gt 1 ]; then
        echo "  E2E TEST SUITE — $num_tests tests × $runs cycles = $total_runs runs"
    else
        echo "  E2E TEST SUITE — $num_tests automated tests"
    fi
    echo "  Effective physics profile: ${E2E_PHYSICS_PROFILE}"
    echo "========================================================================"

    local cycle=0
    while [ "$cycle" -lt "$runs" ]; do
        cycle=$((cycle + 1))
        if [ "$runs" -gt 1 ]; then
            echo ""
            echo "────── Cycle $cycle / $runs ──────"
        fi

        local idx=0
        local test_idx=0
        for entry in "${tests[@]}"; do
            idx=$((idx + 1))
            local name="${entry%%:*}"
            local script="${entry#*:}"
            local test_start
            test_start=$(date +%s)

            if [ "$runs" -gt 1 ]; then
                echo -e "\n━━━ [cycle $cycle, $idx/$num_tests] $name ━━━"
            else
                echo -e "\n━━━ [$idx/$num_tests] $name ━━━"
            fi

            # Per-cycle log naming: append -r<cycle> when runs>1, else legacy
            local log_name="$name"
            if [ "$runs" -gt 1 ]; then
                log_name="${name}-r${cycle}"
            fi

            local rc=0
            run_single_e2e "$script" "$log_name" || rc=$?

            local test_end
            test_end=$(date +%s)
            local duration=$((test_end - test_start))

            # Retry-eligibility check (same patterns as elsewhere)
            local was_retried=0
            if [ -n "$LAST_INFRA_REASON_FIRST" ]; then
                case "$LAST_INFRA_REASON_FIRST" in
                    sitl-died-no-output*|sitl-exited-during-startup*|startup-timeout*|sitl-alive-but-unresponsive*|sitl-unresponsive*|"fatal-init-signature: bind port"*) was_retried=1 ;;
                esac
                if [ "${E2E_NO_RETRY:-0}" = "1" ]; then
                    was_retried=0
                fi
                total_infra_first=$((total_infra_first + 1))
                test_infra_first[$test_idx]=$((${test_infra_first[$test_idx]} + 1))
            fi

            # Annotation: which cycle this result belongs to (only when runs>1)
            local cycle_tag=""
            if [ "$runs" -gt 1 ]; then
                cycle_tag="cycle=${cycle} "
            fi

            case $rc in
                0)  # PASS
                    total_passed=$((total_passed + 1))
                    test_passes[$test_idx]=$((${test_passes[$test_idx]} + 1))
                    if [ $was_retried -eq 1 ]; then
                        total_recovered=$((total_recovered + 1))
                        results+=("  PASS   ${duration}s  ${cycle_tag}${name}  [recovered: ${LAST_INFRA_REASON_FIRST}]")
                    else
                        results+=("  PASS   ${duration}s  ${cycle_tag}${name}")
                    fi
                    ;;
                2)  # INFRA — final
                    total_infra_final=$((total_infra_final + 1))
                    test_infra_final[$test_idx]=$((${test_infra_final[$test_idx]} + 1))
                    if [ $was_retried -eq 1 ]; then
                        results+=("  INFRA  ${duration}s  ${cycle_tag}${name}  [${LAST_INFRA_REASON_FIRST} → STILL INFRA: ${LAST_INFRA_REASON}]")
                    else
                        if [ -z "$LAST_INFRA_REASON_FIRST" ]; then
                            total_infra_first=$((total_infra_first + 1))
                            test_infra_first[$test_idx]=$((${test_infra_first[$test_idx]} + 1))
                        fi
                        results+=("  INFRA  ${duration}s  ${cycle_tag}${name}  [${LAST_INFRA_REASON:-unknown}]")
                    fi
                    ;;
                *)  # CONTROLLER_FAIL
                    total_ctrl_fail=$((total_ctrl_fail + 1))
                    test_ctrl_fails[$test_idx]=$((${test_ctrl_fails[$test_idx]} + 1))
                    if [ $was_retried -eq 1 ]; then
                        total_recovered=$((total_recovered + 1))
                        results+=("  FAIL   ${duration}s  ${cycle_tag}${name}  [retried after ${LAST_INFRA_REASON_FIRST}]")
                    else
                        results+=("  FAIL   ${duration}s  ${cycle_tag}${name}")
                    fi
                    ;;
            esac
            test_idx=$((test_idx + 1))
        done
    done

    local suite_end
    suite_end=$(date +%s)
    local suite_duration=$((suite_end - suite_start))
    local suite_min=$((suite_duration / 60))
    local suite_sec=$((suite_duration % 60))

    local effective_total=$((total_passed + total_ctrl_fail))
    local first_infra_pct=0
    if [ $total_runs -gt 0 ]; then
        first_infra_pct=$(( (total_infra_first * 100) / total_runs ))
    fi

    echo ""
    echo "========================================================================"
    echo "  E2E TEST SUITE RESULTS"
    if [ "$runs" -gt 1 ]; then
        echo "  $num_tests tests × $runs cycles = $total_runs total runs"
    fi
    echo "========================================================================"
    echo "  Total: $total_runs  Passed: $total_passed  Ctrl-fail: $total_ctrl_fail  Infra-fail (final): $total_infra_final  Duration: ${suite_min}m ${suite_sec}s"
    echo "  First-attempt infra: $total_infra_first (${first_infra_pct}%) — $total_recovered recovered by retry, $total_infra_final still infra after retry"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $total_passed/$effective_total (excludes final infra failures)"
    else
        echo "  Effective pass rate: n/a (all runs were infra-fails)"
    fi

    if [ $first_infra_pct -gt $SUITE_INFRA_THRESHOLD_PCT ]; then
        echo ""
        echo -e "  ${YELLOW}⚠ SUITE UNSTABLE — first-attempt infra rate ${first_infra_pct}% > ${SUITE_INFRA_THRESHOLD_PCT}% threshold${NC}"
    fi

    if [ "$runs" -gt 1 ]; then
        echo ""
        echo "  Per-test breakdown (across $runs cycles):"
        printf "    %-22s  %-7s  %-9s  %-11s  %-13s  %s\n" "test" "pass" "ctrl-fail" "infra (1st)" "infra (final)" "flag"
        local i=0
        for name in "${test_names[@]}"; do
            local p="${test_passes[$i]}"
            local cf="${test_ctrl_fails[$i]}"
            local infra1="${test_infra_first[$i]}"
            local infraF="${test_infra_final[$i]}"
            local flag=""
            if [ $infra1 -ge $TEST_FLAKY_THRESHOLD ]; then
                flag="⚠ FLAKY"
            fi
            printf "    %-22s  %d/%d    %d/%d      %d/%d         %d/%d           %s\n" "$name" "$p" "$runs" "$cf" "$runs" "$infra1" "$runs" "$infraF" "$runs" "$flag"
            i=$((i + 1))
        done
    fi

    echo ""
    for r in "${results[@]}"; do
        echo "$r"
    done

    echo ""
    if [ "$runs" -gt 1 ]; then
        echo "  Per-cycle logs preserved (no overwriting):"
        echo "    Attempt 1: /tmp/bf-e2e-<test>-r<N>.log"
        echo "    Attempt 2: /tmp/bf-e2e-<test>-r<N>-attempt2.log (only if retried)"
    else
        echo "  Logs:"
        echo "    Attempt 1: /tmp/bf-e2e-<test>.log"
        echo "    Attempt 2: /tmp/bf-e2e-<test>-attempt2.log (only if retried)"
    fi
    echo "========================================================================"

    # Three-state return: 0 = all pass, 1 = any controller fail, 2 = infra-only failures
    if   [ $total_ctrl_fail -gt 0 ];  then return 1
    elif [ $total_infra_final -gt 0 ]; then return 2
    else                                    return 0
    fi
}

do_e2e() {
    local MODE="${1:-headless}"

    # Verify e2e script exists
    if [ ! -f "$E2E_SCRIPT" ]; then
        fail "E2E test script not found at $E2E_SCRIPT"
    fi

    # Activate venv if not already active
    if [[ "${VIRTUAL_ENV:-}" != *".venv"* ]]; then
        source .venv/bin/activate
    fi
    unset PYTHONPATH 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$PATH"

    # Pre-flight: verify elodin Python module is importable
    if ! python3 -c "import elodin" 2>/dev/null; then
        fail "elodin Python module not installed in .venv. Run './run.sh rebuild-elodin' (inside nix develop) first."
    fi

    if [ "$MODE" = "editor" ]; then
        # Editor mode: use 'elodin editor' for 3D viewport
        ELODIN_BIN="${HOME}/.cargo/bin/elodin"
        if [ ! -x "$ELODIN_BIN" ]; then
            ELODIN_BIN=$(command -v elodin 2>/dev/null) || fail "elodin binary not found. Run './run.sh rebuild-elodin' first."
        fi

        echo -e "\n${GREEN}▶ Running ALT_HOLD E2E test with editor${NC}"
        echo -e "  Script: $E2E_SCRIPT"
        echo -e "  Editor: $ELODIN_BIN"
        echo -e "  Log file: $E2E_LOG_FILE\n"

        PYTHONUNBUFFERED=1 \
        RUST_LOG=elodin_editor=info,impeller2_bevy=info \
            "$ELODIN_BIN" editor "$E2E_SCRIPT" 2>&1 | tee "$E2E_LOG_FILE"
    else
        # Headless mode: run Python directly + manage BF SITL ourselves.
        # This avoids the uv/s10 path issue with scripts outside the
        # elodin project directory.
        ELF="$BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf"
        if [ ! -f "$ELF" ]; then
            fail "Betaflight SITL not found at $ELF. Run './run.sh build-bf' first."
        fi

        # Kill stale BF processes from previous runs
        pkill -f betaflight_SITL 2>/dev/null || true
        sleep 0.2

        echo -e "\n${GREEN}▶ Running E2E test (headless): $E2E_SCRIPT${NC}"
        echo -e "  Betaflight: $ELF"
        echo -e "  Log file: $E2E_LOG_FILE\n"

        # Start Betaflight SITL in background (must run from betaflight dir
        # so it finds eeprom.bin with ARM/ANGLE/ALT_HOLD aux config)
        (cd "$BETAFLIGHT_DIR" && exec "$ELF") &
        BF_PID=$!
        echo -e "  Betaflight SITL started (PID $BF_PID) from $BETAFLIGHT_DIR"

        # Give BF a moment to bind its UDP/TCP ports
        sleep 1

        # Run the E2E test directly with Python (--no-s10: don't use s10)
        PYTHONUNBUFFERED=1 \
            python3 "$E2E_SCRIPT" run --no-s10 2>&1 | tee "$E2E_LOG_FILE"
        TEST_EXIT=$?

        # Cleanup: kill Betaflight SITL
        kill $BF_PID 2>/dev/null
        wait $BF_PID 2>/dev/null

        if [ $TEST_EXIT -ne 0 ]; then
            fail "E2E test exited with code $TEST_EXIT"
        fi
    fi
}

do_check_logs() {
    echo -e "\n${YELLOW}── Log Analysis ──${NC}"
    echo "VTableMsg registrations:"
    grep "VTableMsg registered" "$LOG_FILE" 2>/dev/null || echo "  (none)"
    echo ""
    echo "Cache depth (last 3):"
    grep "cache depth" "$LOG_FILE" 2>/dev/null | tail -3 || echo "  (none)"
    echo ""
    echo "Table received (last 5):"
    grep "Table received" "$LOG_FILE" 2>/dev/null | tail -5 || echo "  (none)"
}

# ── Main ──
MODE="${1:-all}"
case "$MODE" in
    build-bf)
        do_build_bf
        ;;
    rebuild-elodin)
        do_rebuild
        ;;
    run)
        do_run
        ;;
    all)
        do_build_bf
        do_rebuild
        do_run
        ;;
    e2e-all)
        do_e2e_all "${2:-1}"
        ;;
    e2e-angle-althold)
        do_single_test_runs "$E2E_SCRIPT" "angle-althold" "${2:-1}"
        ;;
    e2e-angle-althold-editor)
        do_e2e editor
        ;;
    e2e-acro-althold)
        do_single_test_runs "$E2E_ACRO_SCRIPT" "acro-althold" "${2:-1}"
        ;;
    e2e-acro-althold-editor)
        E2E_SCRIPT="$E2E_ACRO_SCRIPT" do_e2e editor
        ;;
    e2e-flight)
        do_single_test_runs "$E2E_FLIGHT_SCRIPT" "flight" "${2:-1}"
        ;;
    e2e-flight-editor)
        E2E_SCRIPT="$E2E_FLIGHT_SCRIPT" do_e2e editor
        ;;
    e2e-failsafe-althold)
        do_single_test_runs "$E2E_FAILSAFE_SCRIPT" "failsafe-althold" "${2:-1}"
        ;;
    e2e-failsafe-althold-editor)
        E2E_SCRIPT="$E2E_FAILSAFE_SCRIPT" do_e2e editor
        ;;
    e2e-poshold)
        do_single_test_runs "$E2E_POSHOLD_SCRIPT" "poshold" "${2:-1}"
        ;;
    e2e-poshold-editor)
        E2E_SCRIPT="$E2E_POSHOLD_SCRIPT" do_e2e editor
        ;;
    e2e-nosettle-takeoff)
        do_single_test_runs "$E2E_NOSETTLE_SCRIPT" "nosettle-takeoff" "${2:-1}"
        ;;
    e2e-failsafe-init)
        do_single_test_runs "$E2E_FAILSAFE_INIT_SCRIPT" "failsafe-init" "${2:-1}"
        ;;
    e2e-ground-idle)
        do_single_test_runs "$E2E_GROUND_IDLE_SCRIPT" "ground-idle" "${2:-1}"
        ;;
    e2e-center-semantics)
        do_single_test_runs "$E2E_CENTER_SCRIPT" "center-semantics" "${2:-1}"
        ;;
    e2e-midair-activation)
        do_single_test_runs "$E2E_MIDAIR_SCRIPT" "midair-activation" "${2:-1}"
        ;;
    e2e-smooth-takeoff)
        do_single_test_runs "$E2E_SMOOTH_TAKEOFF_SCRIPT" "smooth-takeoff" "${2:-1}"
        ;;
    e2e-manual-landing-safety)
        do_single_test_runs "$E2E_MANUAL_LANDING_SAFETY_SCRIPT" "manual-landing-safety" "${2:-1}"
        ;;
    e2e-low-alt-horizontal)
        do_single_test_runs "$E2E_LOW_ALT_HORIZONTAL_SCRIPT" "low-alt-horizontal" "${2:-1}"
        ;;
    e2e-all-strict)
        # 12 tests (full set) under strict physics profile. Diagnostic lane —
        # may show new failures under stricter IGE (use as non-blocking).
        export E2E_PHYSICS_PROFILE=strict
        do_e2e_all "${2:-1}"
        ;;
    e2e-focused)
        # IGE-sensitive focused 6 tests. Defaults to 5 cycles for confidence.
        do_focused_suite "${E2E_PHYSICS_PROFILE}" "${2:-5}"
        ;;
    e2e-wind-althold)
        do_wind_suite WIND_TESTS "${2:-1}" "ALTHOLD"
        ;;
    e2e-wind-poshold)
        do_wind_suite WIND_POSHOLD_TESTS "${2:-1}" "POSHOLD"
        ;;
    e2e-strict|e2e-realistic)
        # v10.4: removed. Migration error.
        echo "ERROR: '$1' was removed in v10.4." >&2
        echo "" >&2
        if [ "$1" = "e2e-strict" ]; then
            echo "  - For ALL 12 tests under strict physics: use 'e2e-all-strict'" >&2
            echo "  - For IGE-sensitive focused 6 under strict:" >&2
            echo "      E2E_PHYSICS_PROFILE=strict ./run.sh e2e-focused" >&2
        else
            echo "  - For ALL 12 tests under realistic (now default): use 'e2e-all'" >&2
            echo "  - For IGE-sensitive focused 6 under realistic: use 'e2e-focused'" >&2
        fi
        echo "" >&2
        echo "  realistic is now the default physics profile for all targets." >&2
        echo "  baseline remains available via env var override for ad-hoc diagnostic." >&2
        exit 1
        ;;
    check)
        do_check_logs
        ;;
    *)
        echo "Usage: ./run.sh [build-bf|rebuild-elodin|run|e2e-all|e2e-*|all|check]"
        echo ""
        echo "Build / run:"
        echo "  build-bf                              - clean + build betaflight SITL .elf"
        echo "  rebuild-elodin                        - rebuild elodin (Python SDK + editor binary)"
        echo "  run                                   - run editor (skip rebuild)"
        echo "  all                                   - build-bf + rebuild-elodin + run (default)"
        echo "  check                                 - analyze log file at $LOG_FILE"
        echo ""
        echo "E2E suites (all under realistic IGE physics by default):"
        echo "  e2e-all          [N|--runs=N]         - ALL 12 automated tests (default 1 cycle)"
        echo "  e2e-all-strict   [N|--runs=N]         - ALL 12 tests under strict IGE (diagnostic lane)"
        echo "  e2e-focused      [N|--runs=N]         - 6 IGE-sensitive tests (default 5 cycles)"
        echo "  e2e-wind-althold [N|--runs=N]         - 6 ALTHOLD tests under wind (default moderate)"
        echo "  e2e-wind-poshold [N|--runs=N]         - POSHOLD wind drift test"
        echo ""
        echo "Single-test targets (all under realistic by default; accept [N|--runs=N]):"
        echo "  e2e-ground-idle           [N|--runs=N] - ground idle regression test"
        echo "  e2e-smooth-takeoff        [N|--runs=N] - smooth takeoff ramp test"
        echo "  e2e-manual-landing-safety [N|--runs=N] - manual touchdown anti-regression (low-pass + commit-land)"
        echo "  e2e-center-semantics      [N|--runs=N] - center-stick semantics test"
        echo "  e2e-nosettle-takeoff      [N|--runs=N] - no-settle takeoff FSM test"
        echo "  e2e-angle-althold         [N|--runs=N] - ANGLE+ALTHOLD flight cycle test"
        echo "  e2e-acro-althold          [N|--runs=N] - ACRO+ALTHOLD flight cycle test"
        echo "  e2e-flight                [N|--runs=N] - horizontal flight test"
        echo "  e2e-failsafe-althold      [N|--runs=N] - failsafe landing test"
        echo "  e2e-failsafe-init         [N|--runs=N] - failsafe from INITIALIZE test"
        echo "  e2e-midair-activation     [N|--runs=N] - mid-air ALTHOLD activation safety test"
        echo "  e2e-poshold               [N|--runs=N] - position hold test"
        echo "  e2e-low-alt-horizontal    [N|--runs=N] - low-altitude horizontal flight (spin-lock regression)"
        echo ""
        echo "  e2e-*-editor                          - any test above with 3D viewport (no multi-run)"
        echo ""
        echo "Environment variables (override defaults):"
        echo "  E2E_PHYSICS_PROFILE=baseline|strict|realistic (default: realistic)"
        echo "                                        - IGE physics severity. baseline retained for ad-hoc"
        echo "                                          diagnostic but no built-in target uses it."
        echo "  E2E_WIND_PROFILE=calm|light|moderate|strong|gusty (default: calm)"
        echo "                                        - ambient wind, gusty matches hardware test (2-5 m/s, 1-2s gusts)"
        echo "  Both can be set independently. e2e-wind-* targets default wind to 'moderate' if unset."
        echo ""
        echo "Removed targets (v10.4):"
        echo "  e2e-strict          → use 'e2e-all-strict' or E2E_PHYSICS_PROFILE=strict ./run.sh e2e-focused"
        echo "  e2e-realistic       → use 'e2e-all' (realistic is now default) or 'e2e-focused'"
        exit 1
        ;;
esac
