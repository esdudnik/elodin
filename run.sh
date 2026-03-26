#!/usr/bin/env bash
set -euo pipefail

# ── Elodin + Betaflight build & run script ──
# Run from the elodin repo root, inside `nix develop` shell.
# Usage: ./run.sh [build-bf|rebuild-elodin|run|e2e-althold|e2e-althold-editor|all|check]
#   build-bf              - clean + build betaflight SITL .elf
#   rebuild-elodin        - rebuild elodin (Python SDK + editor binary)
#   run                   - just run the editor (skip rebuild)
#   e2e-althold           - run ALT_HOLD E2E test (headless)
#   e2e-althold-editor    - run ALT_HOLD E2E test (with 3D viewport)
#   all                   - build-bf + rebuild-elodin + run (default)
#   check                 - analyze log file

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EXAMPLE="examples/betaflight-sitl/main.py"
E2E_SCRIPT="$REPO_ROOT/e2e_althold_test.py"
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
    e2e-althold)
        do_e2e headless
        ;;
    e2e-althold-editor)
        do_e2e editor
        ;;
    check)
        do_check_logs
        ;;
    *)
        echo "Usage: ./run.sh [build-bf|rebuild-elodin|run|e2e-althold|e2e-althold-editor|all|check]"
        echo "  build-bf              - clean + build betaflight SITL .elf"
        echo "  rebuild-elodin        - rebuild elodin (Python SDK + editor binary)"
        echo "  run                   - run editor (skip rebuild)"
        echo "  e2e-althold           - run ALT_HOLD E2E test (headless)"
        echo "  e2e-althold-editor    - run ALT_HOLD E2E test (with 3D viewport)"
        echo "  all                   - build-bf + rebuild-elodin + run (default)"
        echo "  check                 - analyze log file at $LOG_FILE"
        exit 1
        ;;
esac
