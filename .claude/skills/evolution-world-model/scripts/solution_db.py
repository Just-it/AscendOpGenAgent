#!/usr/bin/env python3
"""
solution_db.py — JSONL-based lineage tracker for CAKE3 evolution.

Each record in the JSONL file represents one evaluated solution variant.
Records are appended in order; the file grows monotonically (no in-place edits).

Schema (one JSON object per line):
{
  "solution_id": "round_1_parallel_2",
  "node_id": "n3",
  "parent_solution_id": null,
  "speedup": 1.19,
  "kernel_path": "round_1/parallel_2/{op_name}Custom/op_kernel/{op_name}_custom.cpp",
  "strategy_combination": ["P1", "P4"],
  "round": 1,
  "op_name": "Softmax",
  "evo_timestamp": "20260309_093828"
}
"""

import json
import os
from typing import Optional


class SolutionDB:
    """
    JSONL-based lineage tracker for evolution solution variants.

    Usage:
        db = SolutionDB("output/FastGELU_evo_20260309/solution_db.jsonl")
        db.add({
            "solution_id": "round_1_parallel_0",
            "node_id": "n1",
            "parent_solution_id": None,
            "speedup": 1.35,
            "kernel_path": "round_1/parallel_0/FastGELUCustom/op_kernel/FastGELU_custom.cpp",
            "strategy_combination": ["P1", "P7"],
            "round": 1,
            "op_name": "FastGELU",
            "evo_timestamp": "20260309_093828",
        })
        best3 = db.get_best(n=3)
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the SolutionDB.

        Args:
            db_path: Path to the JSONL file. Created on first write if absent.
        """
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, record: dict) -> None:
        """
        Append a solution record to the JSONL database.

        The record is appended as a single JSON line. Required fields:
          - solution_id (str): unique identifier, e.g. "round_1_parallel_2"
          - node_id (str): world model node ID
          - speedup (float | None): measured speedup vs baseline
          - kernel_path (str): relative path to the compiled kernel .cpp
          - strategy_combination (list[str]): list of strategy IDs
          - round (int): evolution round number
          - op_name (str): operator name
          - evo_timestamp (str): timestamp of the evolution run

        Optional fields:
          - parent_solution_id (str | None): solution_id of the parent variant

        Args:
            record: Dict containing the record fields above.

        Raises:
            OSError: If the file cannot be opened for appending.
        """
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _load_all(self) -> list[dict]:
        """Load all records from the JSONL file. Returns [] if file absent."""
        if not os.path.exists(self.db_path):
            return []
        records = []
        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
        return records

    def get(self, solution_id: str) -> Optional[dict]:
        """
        Retrieve a record by solution_id.

        Args:
            solution_id: The unique solution identifier.

        Returns:
            The matching record dict, or None if not found.
        """
        for record in self._load_all():
            if record.get("solution_id") == solution_id:
                return record
        return None

    def get_best(self, n: int = 3) -> list[dict]:
        """
        Return the top-N records by speedup (descending).

        Records without a speedup value (None or missing) are excluded.

        Args:
            n: Number of top records to return.

        Returns:
            List of up to n record dicts, sorted by speedup descending.
        """
        records = self._load_all()
        valid = [r for r in records if r.get("speedup") is not None]
        valid.sort(key=lambda r: r["speedup"], reverse=True)
        return valid[:n]

    def get_by_node(self, node_id: str) -> list[dict]:
        """
        Return all records for a given world model node_id.

        Args:
            node_id: The node identifier to filter by.

        Returns:
            List of matching record dicts.
        """
        return [r for r in self._load_all() if r.get("node_id") == node_id]

    def count(self) -> int:
        """Return the total number of records in the database."""
        return len(self._load_all())


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_solution_id(round_num: int, parallel_idx: int) -> str:
    """Generate a canonical solution_id from round and parallel index."""
    return f"round_{round_num}_parallel_{parallel_idx}"


def make_kernel_path(op_name: str, round_num: int, parallel_idx: int) -> str:
    """Generate the relative kernel path for a solution."""
    return (
        f"round_{round_num}/parallel_{parallel_idx}/"
        f"{op_name}Custom/op_kernel/{op_name}_custom.cpp"
    )
