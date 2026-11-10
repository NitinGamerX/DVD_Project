#!/usr/bin/env bash
# =============================================================================
# One-time environment setup for the DVD ML leakage-estimation project.
#   ./setup.sh               system packages + Python venv + checks
#   ./setup.sh --with-latex  also install LaTeX for the report (~700 MB)
#   ./setup.sh --no-apt      skip apt (packages already installed / no sudo)
# Safe to re-run.
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

WITH_LATEX=0; NO_APT=0
for arg in "$@"; do
  case "$arg" in
    --with-latex) WITH_LATEX=1 ;;
    --no-apt)     NO_APT=1 ;;
    -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }

# ---- preflight ---------------------------------------------------------------
say "Preflight"
. /etc/os-release
echo "OS: $PRETTY_NAME   kernel: $(uname -r)"
grep -qi microsoft /proc/version && echo "running under WSL" || warn "not WSL (fine on native Linux)"
case "$PWD" in
  /mnt/*) warn "project is on the Windows drive ($PWD). ngspice runs and file I/O will be" \
               "much slower; move it under your Linux home (~/dvd_project)." ;;
esac
[ "${VERSION_ID:-0}" = "24.04" ] || warn "tested on Ubuntu 24.04; you have ${VERSION_ID:-unknown}."

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ---- system packages -----------------------------------------------------------
APT_PKGS="ngspice python3-venv python3-pip make git"
LATEX_PKGS="texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science latexmk"
if [ "$NO_APT" -eq 0 ]; then
  say "Installing system packages: $APT_PKGS"
  $SUDO apt-get update -y
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends $APT_PKGS
  if [ "$WITH_LATEX" -eq 1 ]; then
    say "Installing LaTeX: $LATEX_PKGS"
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends $LATEX_PKGS
  fi
fi

command -v ngspice >/dev/null || die "ngspice not installed (re-run without --no-apt)."
echo "ngspice: $(ngspice --version | grep -o 'ngspice-[0-9.]*' | head -1)"

# ---- Python --------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' || die \
  "$("$PYTHON" --version) found; the pinned packages need Python >= 3.12 (Ubuntu 24.04 default).
       Either upgrade the WSL distro to 24.04, or install python3.12 + python3.12-venv
       and re-run as:  PYTHON=python3.12 ./setup.sh"

if [ ! -x .venv/bin/python ]; then
  say "Creating virtual environment .venv"
  "$PYTHON" -m venv .venv
fi
say "Installing pinned Python packages (requirements.txt)"
.venv/bin/python -m pip install --disable-pip-version-check -q --only-binary=:all: -r requirements.txt

# ---- git -----------------------------------------------------------------------
if [ ! -d .git ]; then
  say "Initialising git repository"
  git init -q -b main
  git add -A
  if git config user.email >/dev/null && git config user.name >/dev/null; then
    git commit -q -m "Project scaffold: config, model file, environment check" && echo "initial commit created"
  else
    warn "git user.name/user.email not set; skipped the initial commit. Set them, then:
         git commit -m 'Project scaffold'"
  fi
fi

# ---- verification ----------------------------------------------------------------
say "Environment check (scripts/00_check_env.py)"
.venv/bin/python scripts/00_check_env.py

say "Done. Open the project with:  code ."
