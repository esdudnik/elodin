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
BETAFLIGHT_DIR="$REPO_ROOT/../betaflight"
LOG_FILE="/tmp/bf-elodin.log"
E2E_LOG_FILE="/tmp/bf-e2e.log"

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

# Set by run_single_e2e to the human-readable infra-fail reason whenever it
# returns RC_INFRA_FAIL. Cleared at the start of each call. Callers append
# this to their per-test summary line for at-a-glance diagnosis.
LAST_INFRA_REASON=""

# ── Wait for BF SITL ports to be free ──
# Returns 0 if all ports free within timeout, 1 otherwise.
# On timeout, prints which ports are still held and by which PIDs.
# Args: $1 = timeout in seconds (default 10)
wait_ports_free() {
    local timeout_s="${1:-10}"
    local ports=(5761 9001 9002 9003 9004 2240)
    local start; start=$(date +%s)
    while true; do
        local held=()
        local p
        for p in "${ports[@]}"; do
            local pids
            pids=$(lsof -ti :$p 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            [ -n "$pids" ] && held+=("$p(pids=$pids)")
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
    fi
    if ! kill -0 "$bf_pid" 2>/dev/null; then
        if [ ! -s "$log" ]; then echo "sitl-dead-no-output"
        else echo "sitl-died-during-test"
        fi
        return 0
    fi
    echo ""
}

# ── Headless E2E runner (single test) ──
# Extracted from do_e2e headless path. Used by both individual e2e-* commands
# and do_e2e_all(). Returns one of:
#   $RC_PASS (0)             — test passed
#   $RC_CONTROLLER_FAIL (1)  — test ran, BF/test reported failure
#   $RC_INFRA_FAIL (2)       — BF SITL didn't start cleanly or crashed; not a controller fault
run_single_e2e() {
    local test_script="$1"
    local test_name="$2"
    local log_file="/tmp/bf-e2e-${test_name}.log"
    LAST_INFRA_REASON=""   # cleared per call; set before any RC_INFRA_FAIL return

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
    if ! wait_ports_free 10; then
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

# ── Focused suite for strict/realistic physics profile investigations ──
# Used to reproduce real-hardware "floating" behavior under realistic IGE.
# Tests are the ones strict_test.md identified as IGE-sensitive.
FOCUSED_TESTS=(
    "ground-idle:$E2E_GROUND_IDLE_SCRIPT"
    "nosettle-takeoff:$E2E_NOSETTLE_SCRIPT"
    "failsafe-althold:$E2E_FAILSAFE_SCRIPT"
    "failsafe-init:$E2E_FAILSAFE_INIT_SCRIPT"
    "midair-activation:$E2E_MIDAIR_SCRIPT"
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
    local total_infra=0

    # Per-test counters (parallel arrays — bash 3.2 compat, no associative arrays)
    local test_names=()
    local test_passes=()
    local test_ctrl_fails=()
    local test_infra_fails=()
    for entry in "${FOCUSED_TESTS[@]}"; do
        test_names+=("${entry%%:*}")
        test_passes+=(0)
        test_ctrl_fails+=(0)
        test_infra_fails+=(0)
    done

    # Track each infra-fail occurrence as "<name> r<run_idx>: <reason>" for
    # display after the breakdown table. Lets the reader diagnose the failure
    # mode (ports-still-held, sitl-died-during-test, etc.) without scrolling
    # back through the full run output.
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
            case $rc in
                0)  # RC_PASS
                    test_passes[$i]=$((${test_passes[$i]} + 1))
                    total_passed=$((total_passed + 1))
                    ;;
                2)  # RC_INFRA_FAIL
                    test_infra_fails[$i]=$((${test_infra_fails[$i]} + 1))
                    total_infra=$((total_infra + 1))
                    infra_reasons+=("${name} r${run_idx}: ${LAST_INFRA_REASON:-unknown}")
                    ;;
                *)  # RC_CONTROLLER_FAIL or any other non-zero
                    test_ctrl_fails[$i]=$((${test_ctrl_fails[$i]} + 1))
                    total_ctrl_fail=$((total_ctrl_fail + 1))
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

    # Effective pass rate excludes infra failures (they're not the controller's fault)
    local effective_total=$((total_passed + total_ctrl_fail))

    echo ""
    echo "========================================================================"
    echo "  FOCUSED SUITE RESULTS — profile=$profile  runs=$runs"
    echo "========================================================================"
    echo "  Total runs: $total_runs  Pass: $total_passed  Ctrl-fail: $total_ctrl_fail  Infra-fail: $total_infra  Duration: ${suite_min}m ${suite_sec}s"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $total_passed/$effective_total (excludes infra failures)"
    else
        echo "  Effective pass rate: n/a (all runs were infra-fails)"
    fi
    echo ""
    echo "  Per-test breakdown (across $runs runs):"
    printf "    %-22s  %-9s  %-11s  %-11s\n" "test" "pass" "ctrl-fail" "infra-fail"
    local i=0
    for name in "${test_names[@]}"; do
        local p="${test_passes[$i]}"
        local cf="${test_ctrl_fails[$i]}"
        local inf="${test_infra_fails[$i]}"
        printf "    %-22s  %d/%d      %d/%d        %d/%d\n" "$name" "$p" "$runs" "$cf" "$runs" "$inf" "$runs"
        i=$((i + 1))
    done
    if [ ${#infra_reasons[@]} -gt 0 ]; then
        echo ""
        echo "  Infra-fail reasons:"
        local r
        for r in "${infra_reasons[@]}"; do
            echo "    $r"
        done
    fi

    echo ""
    echo "  Per-run logs: /tmp/bf-e2e-${profile}-<test>-r<N>.log"
    echo "========================================================================"

    # Three-state return: 0 = all pass, 1 = any controller fail, 2 = infra-only failures
    if [ $total_ctrl_fail -gt 0 ]; then return 1
    elif [ $total_infra -gt 0 ]; then return 2
    else                              return 0
    fi
}

# ── Run all automated E2E tests sequentially ──
do_e2e_all() {
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
    )

    local total=${#tests[@]}
    local passed=0
    local ctrl_failed=0
    local infra_failed=0
    local results=()
    local suite_start
    suite_start=$(date +%s)

    echo ""
    echo "========================================================================"
    echo "  E2E TEST SUITE — $total automated tests"
    echo "========================================================================"

    local idx=0
    for entry in "${tests[@]}"; do
        idx=$((idx + 1))
        local name="${entry%%:*}"
        local script="${entry#*:}"
        local test_start
        test_start=$(date +%s)

        echo -e "\n━━━ [$idx/$total] $name ━━━"

        # Guarded execution — no set -e cascade
        local rc=0
        run_single_e2e "$script" "$name" || rc=$?

        local test_end
        test_end=$(date +%s)
        local duration=$((test_end - test_start))

        # Three-state classification: PASS / CONTROLLER_FAIL / INFRA_FAIL
        case $rc in
            0)  # RC_PASS
                passed=$((passed + 1))
                results+=("  PASS   ${duration}s  $name")
                ;;
            2)  # RC_INFRA_FAIL — SITL didn't start cleanly or crashed
                infra_failed=$((infra_failed + 1))
                local reason="${LAST_INFRA_REASON:-unknown}"
                results+=("  INFRA  ${duration}s  $name  [${reason}]  (log: /tmp/bf-e2e-${name}.log)")
                ;;
            *)  # RC_CONTROLLER_FAIL or any other non-zero
                ctrl_failed=$((ctrl_failed + 1))
                results+=("  FAIL   ${duration}s  $name  (log: /tmp/bf-e2e-${name}.log)")
                ;;
        esac
    done

    local suite_end
    suite_end=$(date +%s)
    local suite_duration=$((suite_end - suite_start))
    local suite_min=$((suite_duration / 60))
    local suite_sec=$((suite_duration % 60))

    # Effective pass rate excludes infra failures (not the controller's fault)
    local effective_total=$((passed + ctrl_failed))

    echo ""
    echo "========================================================================"
    echo "  E2E TEST SUITE RESULTS"
    echo "========================================================================"
    echo "  Total: $total  Passed: $passed  Ctrl-fail: $ctrl_failed  Infra-fail: $infra_failed  Duration: ${suite_min}m ${suite_sec}s"
    if [ $effective_total -gt 0 ]; then
        echo "  Effective pass rate: $passed/$effective_total (excludes infra failures)"
    else
        echo "  Effective pass rate: n/a (all runs were infra-fails)"
    fi
    echo ""
    for r in "${results[@]}"; do
        echo "$r"
    done
    echo "========================================================================"

    # Three-state return: 0 = all pass, 1 = any controller fail, 2 = infra-only failures
    if   [ $ctrl_failed -gt 0 ];  then return 1
    elif [ $infra_failed -gt 0 ]; then return 2
    else                               return 0
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
        do_e2e_all
        ;;
    e2e-angle-althold)
        run_single_e2e "$E2E_SCRIPT" "angle-althold"
        ;;
    e2e-angle-althold-editor)
        do_e2e editor
        ;;
    e2e-acro-althold)
        run_single_e2e "$E2E_ACRO_SCRIPT" "acro-althold"
        ;;
    e2e-acro-althold-editor)
        E2E_SCRIPT="$E2E_ACRO_SCRIPT" do_e2e editor
        ;;
    e2e-flight)
        run_single_e2e "$E2E_FLIGHT_SCRIPT" "flight"
        ;;
    e2e-flight-editor)
        E2E_SCRIPT="$E2E_FLIGHT_SCRIPT" do_e2e editor
        ;;
    e2e-failsafe-althold)
        run_single_e2e "$E2E_FAILSAFE_SCRIPT" "failsafe-althold"
        ;;
    e2e-failsafe-althold-editor)
        E2E_SCRIPT="$E2E_FAILSAFE_SCRIPT" do_e2e editor
        ;;
    e2e-poshold)
        run_single_e2e "$E2E_POSHOLD_SCRIPT" "poshold"
        ;;
    e2e-poshold-editor)
        E2E_SCRIPT="$E2E_POSHOLD_SCRIPT" do_e2e editor
        ;;
    e2e-nosettle-takeoff)
        run_single_e2e "$E2E_NOSETTLE_SCRIPT" "nosettle-takeoff"
        ;;
    e2e-failsafe-init)
        run_single_e2e "$E2E_FAILSAFE_INIT_SCRIPT" "failsafe-init"
        ;;
    e2e-ground-idle)
        run_single_e2e "$E2E_GROUND_IDLE_SCRIPT" "ground-idle"
        ;;
    e2e-center-semantics)
        run_single_e2e "$E2E_CENTER_SCRIPT" "center-semantics"
        ;;
    e2e-midair-activation)
        run_single_e2e "$E2E_MIDAIR_SCRIPT" "midair-activation"
        ;;
    e2e-smooth-takeoff)
        run_single_e2e "$E2E_SMOOTH_TAKEOFF_SCRIPT" "smooth-takeoff"
        ;;
    e2e-strict)
        do_focused_suite strict "${2:-1}"
        ;;
    e2e-realistic)
        do_focused_suite realistic "${2:-1}"
        ;;
    check)
        do_check_logs
        ;;
    *)
        echo "Usage: ./run.sh [build-bf|rebuild-elodin|run|e2e-all|e2e-*|all|check]"
        echo "  build-bf                  - clean + build betaflight SITL .elf"
        echo "  rebuild-elodin            - rebuild elodin (Python SDK + editor binary)"
        echo "  run                       - run editor (skip rebuild)"
        echo "  e2e-all                   - run ALL automated E2E tests sequentially"
        echo "  e2e-ground-idle           - ground idle regression test"
        echo "  e2e-smooth-takeoff        - smooth takeoff ramp test"
        echo "  e2e-center-semantics      - center-stick semantics test"
        echo "  e2e-nosettle-takeoff      - no-settle takeoff FSM test"
        echo "  e2e-angle-althold         - ANGLE+ALTHOLD flight cycle test"
        echo "  e2e-acro-althold          - ACRO+ALTHOLD flight cycle test"
        echo "  e2e-flight                - horizontal flight test"
        echo "  e2e-failsafe-althold      - failsafe landing test"
        echo "  e2e-failsafe-init         - failsafe from INITIALIZE test"
        echo "  e2e-midair-activation     - mid-air ALTHOLD activation safety test"
        echo "  e2e-poshold               - position hold test"
        echo "  e2e-*-editor              - any test above with 3D viewport (e.g. e2e-angle-althold-editor)"
        echo "  e2e-strict [N|--runs=N]   - focused suite under strict physics profile, N runs"
        echo "  e2e-realistic [N|--runs=N]   - focused suite under realistic physics profile (no thrust gate, matches real hardware), N runs"
        echo "  all                       - build-bf + rebuild-elodin + run (default)"
        echo "  check                     - analyze log file at $LOG_FILE"
        exit 1
        ;;
esac
