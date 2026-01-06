# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
# Functions
eval "$(echo 'cHllbnYgKCkgCnsgCiAgICBsb2NhbCBjb21tYW5kPSR7MTotfTsKICAgIFsgIiQjIiAtZ3QgMCBd
ICYmIHNoaWZ0OwogICAgY2FzZSAiJGNvbW1hbmQiIGluIAogICAgICAgIGFjdGl2YXRlIHwgZGVh
Y3RpdmF0ZSB8IHJlaGFzaCB8IHNoZWxsKQogICAgICAgICAgICBldmFsICIkKHB5ZW52ICJzaC0k
Y29tbWFuZCIgIiRAIikiCiAgICAgICAgOzsKICAgICAgICAqKQogICAgICAgICAgICBjb21tYW5k
IHB5ZW52ICIkY29tbWFuZCIgIiRAIgogICAgICAgIDs7CiAgICBlc2FjCn0K' | base64 -d)" > /dev/null 2>&1
# Shell Options
shopt -u autocd
shopt -u assoc_expand_once
shopt -u cdable_vars
shopt -u cdspell
shopt -u checkhash
shopt -u checkjobs
shopt -s checkwinsize
shopt -s cmdhist
shopt -u compat31
shopt -u compat32
shopt -u compat40
shopt -u compat41
shopt -u compat42
shopt -u compat43
shopt -u compat44
shopt -s complete_fullquote
shopt -u direxpand
shopt -u dirspell
shopt -u dotglob
shopt -u execfail
shopt -u expand_aliases
shopt -u extdebug
shopt -u extglob
shopt -s extquote
shopt -u failglob
shopt -s force_fignore
shopt -s globasciiranges
shopt -u globstar
shopt -u gnu_errfmt
shopt -u histappend
shopt -u histreedit
shopt -u histverify
shopt -s hostcomplete
shopt -u huponexit
shopt -u inherit_errexit
shopt -s interactive_comments
shopt -u lastpipe
shopt -u lithist
shopt -u localvar_inherit
shopt -u localvar_unset
shopt -s login_shell
shopt -u mailwarn
shopt -u no_empty_cmd_completion
shopt -u nocaseglob
shopt -u nocasematch
shopt -u nullglob
shopt -s progcomp
shopt -u progcomp_alias
shopt -s promptvars
shopt -u restricted_shell
shopt -u shift_verbose
shopt -s sourcepath
shopt -u xpg_echo
set -o braceexpand
set -o hashall
set -o interactive-comments
set -o monitor
set -o onecmd
shopt -s expand_aliases
# Aliases
# Check for rg availability
if ! command -v rg >/dev/null 2>&1; then
  alias rg='/workspaces/node_modules_global/lib/node_modules/\@anthropic-ai/claude-code/vendor/ripgrep/x64-linux/rg'
fi
export PATH=/workspaces/upstageailab-ocr-recsys-competition-ocr-2/.venv/bin\:/home/vscode/.pyenv/plugins/pyenv-virtualenv/shims\:/home/vscode/.pyenv/shims\:/home/vscode/.pyenv/bin\:/home/vscode/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand\:/home/vscode/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli\:/home/vscode/.vscode-server/cli/servers/Stable-994fd12f8d3a5aa16f17d42c041e5809167e845a/server/bin/remote-cli\:/home/vscode/.local/bin\:/home/vscode/google-cloud-sdk/bin\:/workspaces/repomix/bin\:/workspaces/node_modules_global/bin\:/home/vscode/.pyenv/plugins/pyenv-virtualenv/shims\:/home/vscode/.pyenv/bin\:/opt/uv/bin\:/home/vscode/.pyenv/bin\:/home/vscode/.local/bin\:/opt/conda/bin\:/opt/pypoetry/bin\:/usr/local/bin\:/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/usr/games\:/usr/local/games\:/snap/bin\:/home/vscode/.vscode-server/extensions/ms-python.debugpy-2025.19.2025121701-linux-x64/bundled/scripts/noConfigScripts
