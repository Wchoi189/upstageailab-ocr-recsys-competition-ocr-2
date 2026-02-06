# === OCR Dev Environment ===
# Clean, fast, and minimal.
[ -z "$PS1" ] && return

# 1. Environment & Path
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 EDITOR=vim
export HISTCONTROL=ignoreboth:ignoredups:ignorespace HISTSIZE=10000

# Path Helper Function
add_path() { [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]] && PATH="$1:$PATH"; }

# Add Paths (Priority Order)
add_path "$HOME/bin"
add_path "$HOME/.local/bin"
add_path "$HOME/.cargo/bin"
add_path "/opt/uv"
add_path "$PWD/AgentQMS/bin"
add_path "/workspaces/node_modules_global/bin"
add_path "/workspaces/repomix/bin"
export PATH

# 2. Python Setup (PyEnv & UV)
export PYENV_ROOT="$HOME/.pyenv"
export UV_LINK_MODE=copy

[ -d "$PYENV_ROOT" ] && add_path "$PYENV_ROOT/bin"
if command -v pyenv >/dev/null; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# Auto-activate .venv when entering directory
uv_auto_activate() {
    if [ -f ".venv/bin/activate" ] && [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
    fi
}
PROMPT_COMMAND="uv_auto_activate"

# 3. The Ultra-Concise Prompt
# Logic: Shows ONLY current dir and an arrow.
# Arrow turns PURPLE if venv is active, GREEN if not.
# No text labels. No git branch clutter.
set_prompt() {
    local BLUE='\[\033[0;34m\]'
    local GREEN='\[\033[0;32m\]'
    local PURPLE='\[\033[0;35m\]'
    local RESET='\[\033[0m\]'

    # Choose color based on Venv status
    if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
        local ARROW_COLOR="$PURPLE"
    else
        local ARROW_COLOR="$GREEN"
    fi

    # PS1: [BlueDir] [ColorArrow]
    PS1="${BLUE}\W ${ARROW_COLOR}❯${RESET} "
}
PROMPT_COMMAND="${PROMPT_COMMAND};set_prompt"

# 4. Aliases
# Core
alias ls='ls --color=auto' ll='ls -alF' la='ls -A' l='ls -CF'
alias ..='cd ..' ...='cd ../..'
alias grep='grep --color=auto'

# Git (Minimal)
alias gs='git status' ga='git add' gc='git commit' gp='git push'
alias gd='git diff' gl='git log --oneline' gco='git checkout'

# Docker
alias dc='docker-compose' dcu='dc up -d' dcd='dc down' dcl='dc logs' dce='dc exec'

# Python / UV / Project
alias python='uv run python'
alias pytest='uv run pytest'
alias uv-sync='uv sync --group dev'
alias run-tests='uv run python -m pytest tests/'
alias train='uv run python runners/train.py'
alias predict='uv run python runners/predict.py'

# 5. Utilities
mkcd() { mkdir -p "$1" && cd "$1"; }

# Compact Extract
extract() {
    if [ -f "$1" ]; then
        case $1 in
            *.tar.bz2)   tar xjf "$1"     ;; *.tar.gz)    tar xzf "$1"     ;;
            *.bz2)       bunzip2 "$1"     ;; *.rar)       unrar e "$1"     ;;
            *.gz)        gunzip "$1"      ;; *.tar)       tar xf "$1"      ;;
            *.tbz2)      tar xjf "$1"     ;; *.tgz)       tar xzf "$1"     ;;
            *.zip)       unzip "$1"       ;; *.Z)         uncompress "$1"  ;;
            *.7z)        7z x "$1"        ;; *)           echo "Unknown format" ;;
        esac
    else echo "File not found"; fi
}

# Cleanup stale vars
[ "$PYENV_VERSION" = "doc-pyenv" ] && unset PYENV_VERSION
[ -n "$VIRTUAL_ENV" ] && [ ! -d "$VIRTUAL_ENV" ] && unset VIRTUAL_ENV

# Enable Completion
if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
fi

# === End of .bashrc ===
export PATH="/workspaces/bin:$PATH"
alias python="uv run python"
alias pytest="uv run pytest"
