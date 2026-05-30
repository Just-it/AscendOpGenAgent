"""utils/ — stateless library modules imported by engine/, hooks/,
phase_machine/, workflow/, task_config/, and batch/.

No CLI entry points live here. `validate_triton_impl.py` is a thin
re-export of the kernel-verifier skill's canonical implementation —
`from utils.validate_triton_impl import validate` resolves through to
`skills/triton/kernel-verifier/scripts/validate_triton_impl.py` so the
two consumers (autoresearch + the skill itself) can never drift.

Nothing in this package mutates state. Splitting them out makes the
dependency direction obvious: utils sits at the bottom of the stack and
never imports from any sibling package.
"""
