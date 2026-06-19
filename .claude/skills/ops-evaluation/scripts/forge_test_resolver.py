#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SEARCH_PATTERNS = (
    "st/**/*.py",
    "tests/st/**/*.py",
    "**/*st*.py",
    "**/test_*.py",
    "tests/benchmark/**/*.py",
    "**/benchmark_*.py",
)


def _normalize_name(value: str) -> str:
    return value.lower().replace("_", "")


def _strip_custom_suffix(value: str) -> str:
    if value.lower().endswith("_custom"):
        return value[: -len("_custom")]
    return value


def _operator_names(custom_op_name: str) -> set[str]:
    normalized = _normalize_name(custom_op_name)
    stripped = _normalize_name(_strip_custom_suffix(custom_op_name))
    result = {normalized, stripped}
    # Also add bare name without common vendor/infra prefixes so that
    # benchmark scripts like benchmark_sparse_flash_attention_gqa.py can
    # match even when the op name carries an ai_infra_ / npu_ prefix.
    for prefix in ("ai_infra_", "npu_", "custom_"):
        if stripped.startswith(prefix):
            core = stripped[len(prefix):]
            if core:
                result.add(core)
            break
    return result


def _is_under_st_directory(candidate: Path) -> bool:
    parts = [part.lower() for part in candidate.parts]
    for index, part in enumerate(parts[:-1]):
        if part == "st":
            return True
        if part == "tests" and index + 1 < len(parts) - 1 and parts[index + 1] == "st":
            return True
    return False


def _is_test_like_filename(candidate: Path) -> bool:
    filename = candidate.name.lower()
    return filename.startswith("test_") or filename.endswith("_test.py")


def _is_benchmark_filename(candidate: Path) -> bool:
    return candidate.name.lower().startswith("benchmark_")


def _is_under_benchmark_directory(candidate: Path) -> bool:
    return "benchmark" in [p.lower() for p in candidate.parts[:-1]]


def _candidate_is_valid(candidate: Path) -> bool:
    if _is_under_st_directory(candidate):
        return True
    if _is_under_benchmark_directory(candidate):
        return True
    if _is_benchmark_filename(candidate):
        return True
    return _is_test_like_filename(candidate)


def _candidate_priority(candidate: Path, operator_names: set[str]) -> tuple[int, int, int, int, int, str]:
    normalized_filename = _normalize_name(candidate.name)
    name_match = 0 if any(name and name in normalized_filename for name in operator_names) else 1
    benchmark_with_name = 0 if (name_match == 0 and (_is_under_benchmark_directory(candidate) or _is_benchmark_filename(candidate))) else 1
    st_directory = 0 if _is_under_st_directory(candidate) else 1
    benchmark = 0 if (_is_under_benchmark_directory(candidate) or _is_benchmark_filename(candidate)) else 1
    eager = 0 if "eager" in candidate.name.lower() else 1
    return (benchmark_with_name, name_match, st_directory, benchmark, eager, str(candidate))


def _selection_reason(candidate: Path, operator_names: set[str]) -> str:
    reasons: list[str] = []
    normalized_filename = _normalize_name(candidate.name)
    if any(name and name in normalized_filename for name in operator_names):
        reasons.append("operator-name-match")
    if _is_under_st_directory(candidate):
        reasons.append("st-directory")
    if _is_under_benchmark_directory(candidate) or _is_benchmark_filename(candidate):
        reasons.append("benchmark")
    if "eager" in candidate.name.lower():
        reasons.append("eager")
    if not reasons:
        reasons.append("lexicographic")
    return "repo_st:" + ",".join(reasons)


def resolve_forge_test_script(
    repo_root,
    op_relative_path,
    custom_op_name,
    generated_test_path,
    generated_test_script,
):
    repo_root_path = Path(repo_root).resolve()
    op_dir = (repo_root_path / op_relative_path).resolve()
    if not op_dir.is_relative_to(repo_root_path):
        raise ValueError(f"Operator path escapes repo root: {op_dir}")
    if not op_dir.is_dir():
        raise FileNotFoundError(f"Operator directory not found: {op_dir}")

    # `source` refers to the origin of the selected test script.
    # `repo_st` covers repository-local ST scripts, including any valid .py file
    # under `st/` or `tests/st/`, plus only test-like files outside those dirs.
    operator_names = _operator_names(custom_op_name)
    candidates = []
    seen = set()
    for pattern in SEARCH_PATTERNS:
        for path in op_dir.glob(pattern):
            if path.is_file() and path.suffix == ".py" and _candidate_is_valid(path) and path not in seen:
                seen.add(path)
                candidates.append(path)

    if candidates:
        selected = sorted(candidates, key=lambda candidate: _candidate_priority(candidate, operator_names))[0]
        return {
            "test_path": str(selected.parent),
            "test_script": selected.name,
            "source": "repo_st",
            "selection_reason": _selection_reason(selected, operator_names),
        }

    generated_path = Path(generated_test_path).resolve() / generated_test_script
    if generated_path.is_file():
        return {
            "test_path": str(generated_path.parent),
            "test_script": generated_path.name,
            "source": "generated",
            "selection_reason": "generated:generated-fallback",
        }

    raise FileNotFoundError(
        f"No forge test script found under {op_dir} and generated fallback missing: {generated_path}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve forge test script for ops evaluation")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--op-relative-path", required=True)
    parser.add_argument("--custom-op-name", required=True)
    parser.add_argument("--generated-test-path", required=True)
    parser.add_argument("--generated-test-script", required=True)
    args = parser.parse_args(argv)

    result = resolve_forge_test_script(
        repo_root=args.repo_root,
        op_relative_path=args.op_relative_path,
        custom_op_name=args.custom_op_name,
        generated_test_path=args.generated_test_path,
        generated_test_script=args.generated_test_script,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
