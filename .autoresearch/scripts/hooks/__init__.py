"""hooks/ — Claude Code PreToolUse / PostToolUse / Stop hooks.

These scripts are launched by the harness via the paths registered in
setups/autoresearch/settings.json; users do not invoke them directly.
Each hook is a thin dispatcher: parse the tool I/O, consult phase_machine,
emit a {decision, reason} JSON or guidance message, exit 0.

`utils.py` here is the shared hook-side I/O helper (read_hook_input,
emit_status, block_decision, ...) — distinct from the top-level utils/
package, which holds stateless libraries imported by the engine.
"""
