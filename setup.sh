#!/usr/bin/env bash
# =============================================================================
# Deterministic Analysis for Analysts — one-shot environment setup
# Works on Ubuntu/Linux and macOS. Creates a self-contained .venv in this repo,
# installs the pinned requirements, registers a Jupyter kernel, and runs a
# smoke test that fails fast if anything is wrong.
#
#   Usage:  ./setup.sh          (from the new_workshop/ folder)
#           bash setup.sh
# =============================================================================
set -euo pipefail

# --- locate the repo root (this script lives at its top level) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PY_MIN_MINOR=11   # numpy/pandas pins need Python >= 3.11
PY_MAX_MINOR=13   # ... and have wheels through 3.13

# --- pretty output -----------------------------------------------------------
if [ -t 1 ]; then BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; NC=$'\033[0m'
else BOLD=""; GREEN=""; RED=""; YEL=""; NC=""; fi
say()  { printf '%s\n' "${BOLD}▸ $*${NC}"; }
ok()   { printf '%s\n' "${GREEN}✅ $*${NC}"; }
warn() { printf '%s\n' "${YEL}⚠️  $*${NC}"; }
die()  { printf '%s\n' "${RED}❌ $*${NC}" >&2; exit 1; }

OS="$(uname -s)"
say "Platform: $OS"

# --- 1. find a suitable python3 ---------------------------------------------
find_python() {
  local cand
  for cand in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      "$cand" - <<'PY' >/dev/null 2>&1 || continue
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 11) and sys.version_info[:2] <= (3, 13) else 1)
PY
      echo "$cand"; return 0
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [ -z "${PY:-}" ]; then
  echo
  die "Need Python ${PY_MIN_MINOR}–${PY_MAX_MINOR} (3.${PY_MIN_MINOR}..3.${PY_MAX_MINOR}), none found.
    macOS:  brew install python@3.12
    Ubuntu: sudo apt update && sudo apt install -y python3.12 python3.12-venv"
fi
PY_VER="$("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
ok "Using $PY (Python $PY_VER)"

# --- 2. ensure the venv module works (Ubuntu often ships it separately) ------
if ! "$PY" -c 'import venv' >/dev/null 2>&1; then
  die "The 'venv' module is missing.
    Ubuntu: sudo apt install -y python3-venv
    (on macOS venv ships with python3 — reinstall Python if this fails)"
fi

# --- 3. create the virtual environment --------------------------------------
if [ -d "$VENV_DIR" ]; then
  warn "Existing .venv found — reusing it (delete .venv to rebuild from scratch)."
else
  say "Creating virtual environment at .venv"
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
VPY="$VENV_DIR/bin/python"
[ -x "$VPY" ] || die "venv python not found at $VPY"

# --- 4. install dependencies -------------------------------------------------
say "Upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip
say "Installing requirements.txt (this can take a minute)"
"$VPY" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
ok "Dependencies installed"

# --- 5. register a Jupyter kernel -------------------------------------------
say "Registering Jupyter kernel 'ni-workshop'"
"$VPY" -m ipykernel install --user --name ni-workshop \
  --display-name "Python (NI Workshop)" >/dev/null 2>&1 \
  && ok "Kernel registered" || warn "Could not register kernel (not fatal)"

# --- 6. smoke test: imports, data, and determinism --------------------------
say "Smoke test — imports & data"
"$VPY" - <<'PY' || die "Smoke test failed: scientific stack or data not loading."
import importlib, sys, pathlib
for m in ("numpy","pandas","scipy","matplotlib","seaborn"):
    importlib.import_module(m)
root = pathlib.Path.cwd()
for f in ("data/online_banking_visit_clickouts.csv","data/online_banking_daily_cost.csv","data/visits.csv"):
    assert (root / f).exists(), f"missing data file: {f}"
print("  imports OK · data files present")
PY

say "Smoke test — determinism (profile-data run twice must match byte-for-byte)"
R1="$("$VPY" .claude/skills/profile-data/profile.py --metric revenue --grain click --slice channel=Bing 2>/dev/null)"
R2="$("$VPY" .claude/skills/profile-data/profile.py --metric revenue --grain click --slice channel=Bing 2>/dev/null)"
if [ "$R1" = "$R2" ] && [ -n "$R1" ]; then
  ok "Deterministic: two runs are byte-identical"
else
  die "Determinism check failed — two runs differed (or produced no output)."
fi

# --- done --------------------------------------------------------------------
echo
ok "Setup complete."
cat <<EOF

${BOLD}Next steps${NC}
  1. Activate the environment:
       source .venv/bin/activate            # macOS / Linux
  2. Launch the notebooks:
       jupyter lab                          # then open notebooks/00 → 07 in order
     (pick the "Python (NI Workshop)" kernel if prompted)
  3. Or run a skill directly, e.g.:
       python .claude/skills/decide/decide.py \\
         --a "Summit Direct Business" --b "Cedar Business Bank" --slice channel=Bing

Open this folder ($(basename "$SCRIPT_DIR")/) as the project root in VS Code / Cursor
so Claude Code discovers the skills in .claude/skills/.
EOF
