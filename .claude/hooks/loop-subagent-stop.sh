#!/usr/bin/env bash
# loop-subagent-stop.sh — SubagentStop hook.
#
# Triggered every time a subagent (spawned via Task tool) terminates.
# Validates that evolution partial subagents (ops-partial)
# actually executed the work they claim, by auditing their transcript JSONL.
#
# Routing by agent_type (from stdin JSON):
#   ops-partial)
#   *                              →  allow (no enforcement)
#
# Audit relies on `transcript_audit.py` which inspects the subagent's full
# tool-call history (cannot be lied about — Claude Code's own log).
#
# Escape hatch: LOOP_HOOK_DISABLE=1 downgrades all blocks to warn-only.

set -uo pipefail

HOOK_NAME="loop-subagent-stop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

read_stdin_json

# Extract identifying fields from SubagentStop stdin JSON
agent_type="$(json_get "agent_type")"
agent_transcript="$(json_get "agent_transcript_path")"
last_msg="$(json_get "last_assistant_message")"
cwd_field="$(json_get "cwd")"
stop_hook_active="$(json_get "stop_hook_active")"

# Defensive: stop_hook_active=true means we already blocked once on the same
# subagent; allow on second visit to avoid infinite loops.
if [[ "${stop_hook_active}" == "True" || "${stop_hook_active}" == "true" ]]; then
    hook_allow
fi

# Resolve agent_type via meta.json if stdin's field is empty.
# Each subagent transcript has a sibling <transcript>.meta.json with authoritative agentType.
if [[ -z "${agent_type}" && -n "${agent_transcript}" ]]; then
    meta_path="${agent_transcript%.jsonl}.meta.json"
    if [[ -f "${meta_path}" ]]; then
        agent_type="$(python3 -c "
import json, sys
try:
    m = json.load(open('${meta_path}'))
    print(m.get('agentType', ''))
except Exception:
    pass
" 2>/dev/null)"
    fi
fi

# Route: only enforce on evolution partials. Other subagent types pass through.
case "${agent_type}" in
    ops-partial)
        : # fall through to strict audit
        ;;
    *)
        hook_allow
        ;;
esac

# Transcript missing → can't audit; fail open (main Stop hook + filesystem
# evidence will catch downstream problems via existing R-rules).
if [[ -z "${agent_transcript}" ]] || [[ ! -f "${agent_transcript}" ]]; then
    hook_log "transcript missing/empty for ${agent_type} subagent; skipping audit"
    hook_allow
fi

# Run the transcript audit. transcript_audit.py prints violations to stderr
# (one per line) and exits 1 if any.
audit_py="${PROJECT_ROOT}/.claude/skills/evolution-world-model/scripts/transcript_audit.py"
if [[ ! -f "${audit_py}" ]]; then
    hook_log "transcript_audit.py not found at ${audit_py}; skipping audit"
    hook_allow
fi

audit_output="$(python3 "${audit_py}" audit-partial \
    --transcript "${agent_transcript}" \
    --partial-type "${agent_type}" \
    --last-msg "${last_msg:-}" \
    --cwd "${cwd_field:-${PWD}}" 2>&1 1>/dev/null)"
audit_rc=$?

if [[ ${audit_rc} -ne 0 && -n "${audit_output}" ]]; then
    hook_block "SubagentStop audit failed for ${agent_type}:
${audit_output}

Hint: this subagent's transcript shows it claimed completion but did not
actually invoke the required evaluation/build scripts. Inspect transcript at:
  ${agent_transcript}
To bypass in emergency only: export LOOP_HOOK_DISABLE=1"
fi

hook_allow
