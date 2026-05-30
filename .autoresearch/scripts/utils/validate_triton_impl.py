"""Re-export of the kernel-verifier skill's validate_triton_impl.

Single source of truth lives at:
    skills/triton/kernel-verifier/scripts/validate_triton_impl.py

That script is consumed in two places:
  - This module (autoresearch's quick_check + batch/verify).
  - Claude reading the kernel-verifier skill at runtime — the skill ships
    the canonical file as part of its kit.

Keeping one source prevents the AST regression checker from drifting
between the two consumers (a bug fix here would otherwise have to be
ported by hand into the skill).

Layout assumption: this file lives at .autoresearch/scripts/utils/, and
skills/ is a sibling of .autoresearch/ at the repo root. If either tree
moves, fix `_SKILL_SCRIPTS` below.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../utils -> .../scripts -> .../.autoresearch -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SKILL_SCRIPTS = os.path.join(
    _REPO_ROOT, "skills", "triton", "kernel-verifier", "scripts"
)
if _SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, _SKILL_SCRIPTS)

# Star-import re-exports public names + module-level constants. The explicit
# re-imports below are insurance for the names autoresearch's callers reach
# for by name (`from utils.validate_triton_impl import validate as ...`).
from validate_triton_impl import *  # noqa: F401, F403
from validate_triton_impl import (  # noqa: F401
    validate,
    ALLOWED_TORCH_FUNCS,
    ALLOWED_TENSOR_METHODS,
    ALLOWED_TRITON_ATTRS,
    FORBIDDEN_TENSOR_METHODS,
    FORBIDDEN_PYTHON_STMTS,
)
