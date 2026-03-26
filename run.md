# Running Betaflight SITL with Elodin Editor (macOS ARM)

## Prerequisites (one-time setup)

```bash
cd /path/to/elodin

# 1. Delete any old venv (if exists)
rm -rf .venv

# 2. Enter nix shell
nix develop

# 3. Source shellrc (for install-elodin function)
source $NIX_SHELLRC

# 4. Build everything (Python SDK + elodin-db + elodin editor)
install-elodin

# 5. Build Betaflight SITL binary (if not already built)
cd examples/betaflight-sitl
./build.sh
cd ../..
```

## Running the editor

Every time you open a new terminal:

```bash
cd /path/to/elodin
nix develop
source $NIX_SHELLRC
source .venv/bin/activate
export PATH="$HOME/.cargo/bin:$PATH"
unset PYTHONPATH
elodin editor examples/betaflight-sitl/main.py
```

## Stopping the editor

- **Cmd+Q** or close the window
- Or **Ctrl+C** in the terminal

## Notes

- The `.venv` must be Python 3.13 (created by `install-elodin` inside nix develop). Do not mix with Homebrew Python.
- `unset PYTHONPATH` is required to prevent nix's numpy from conflicting with the venv's numpy.
- The simulation uses `backend="jax"` (IREE backend does not support all JAX features used here).
- Betaflight SITL binary is at `examples/betaflight-sitl/betaflight/obj/main/betaflight_SITL.elf` and is launched automatically by the editor via s10.
- First-time Betaflight config (ARM switch etc.) should already be saved in `eeprom.bin`.
