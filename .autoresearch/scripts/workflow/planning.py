"""PlanStore — single owner of plan.md parsing, rendering, settlement.

Before this module, plan.md was touched from two scripts that didn't
share code:

  - create_plan.py  : parse XML input, validate, allocate pids, render
                      a fresh plan.md preserving the Settled History
                      table.
  - settle.py       : find (ACTIVE) item, mark [x] with KEEP/DISCARD/
                      FAIL tag, advance ACTIVE marker, append to the
                      Settled History table.

Each file had its own line-walker. The render format and the parse
format had to agree by convention; any drift was a silent bug class.
PlanStore puts both behind one type:

  ps = PlanStore(task_dir)
  ps.parse_settled_history()   # carries forward across re-plans
  ps.write(version, item_ids, items, settled_rows)
  settled_id, settled_desc = ps.settle_active(decision, metric)
  ps.allocate_pids(n, progress.next_pid)

Validators (`get_plan_items`, `_PLAN_ITEM_RE`, `is_settled_table_header`)
stay in phase_machine.validators - they're stateless predicates that
both this store and external readers (dashboard, hooks) rely on.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    _PLAN_ITEM_RE, get_plan_items, is_settled_table_header, plan_path,
)


class PlanStore:
    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.path = plan_path(task_dir)

    # ---- read paths -----------------------------------------------------
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def read_text(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_settled_history(self) -> str:
        """Return the Settled History table rows (raw lines, joined with '\n').

        Used by `write()` to carry the history forward when REPLAN rebuilds
        plan.md. Any row drop here would lose the audit chain.
        """
        if not self.exists():
            return ""
        rows: list[str] = []
        in_table = False
        for line in self.read_text().split("\n"):
            stripped = line.strip()
            if is_settled_table_header(line):
                in_table = True
                continue
            if in_table and stripped.startswith("|---"):
                continue
            if in_table and stripped.startswith("|"):
                rows.append(line)
            elif in_table:
                in_table = False
        return ("\n".join(rows) + "\n") if rows else ""

    def parse_pending(self) -> list:
        """Return [{id, description}] for items still [ ] in current plan."""
        return [
            {"id": it["id"], "description": it["description"]}
            for it in get_plan_items(self.task_dir)
            if not it["done"]
        ]

    # ---- pid allocation -----------------------------------------------
    def compute_next_pid(self, progress_next_pid: Optional[int]) -> int:
        """Monotonic pid counter. Trusts the persisted value when present;
        otherwise scans plan.md for the highest pN seen and continues from
        there (back-compat for old tasks that predate the counter)."""
        if progress_next_pid:
            return progress_next_pid
        n = 1
        if self.exists():
            for m in re.finditer(r"\*\*p(\d+)\*\*", self.read_text()):
                n = max(n, int(m.group(1)) + 1)
        return n

    @staticmethod
    def allocate_ids(n_items: int, next_pid: int) -> tuple:
        ids = [f"p{next_pid + i}" for i in range(n_items)]
        return ids, next_pid + n_items

    # ---- rendering ------------------------------------------------------
    @staticmethod
    def render(version: int, item_ids: list, items: list,
               settled_rows: str) -> str:
        lines = [f"# Plan v{version}", "", "## Active Items"]
        for i, (item, pid) in enumerate(zip(items, item_ids)):
            marker = " (ACTIVE)" if i == 0 else ""
            lines.append(f"- [ ] **{pid}**{marker}: {item['desc'].strip()}")
            lines.append(f"  - rationale: {item['rationale'].strip()}")
        lines.append("")
        lines.append("## Settled History")
        lines.append("| Item | Outcome | Metric | Reason |")
        lines.append("|------|---------|--------|--------|")
        if settled_rows:
            lines.append(settled_rows.rstrip())
        return "\n".join(lines) + "\n"

    def write(self, version: int, item_ids: list, items: list,
              settled_rows: str) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.render(version, item_ids, items, settled_rows))

    # ---- settlement -----------------------------------------------------
    def settle_active(self, decision: str,
                      metric: Optional[float]) -> tuple:
        """Mark the (ACTIVE) item as settled, advance ACTIVE to next pending,
        and append a row to the Settled History table.

        Returns (settled_item_id, settled_item_desc). Raises RuntimeError
        when no (ACTIVE) item is present (caller should treat this as a
        plan-corruption signal, same shape as before this refactor).
        """
        if not self.exists():
            raise RuntimeError("plan.md not found")

        if decision == "KEEP" and metric is not None:
            tag = f"[KEEP, metric={metric:.1f}]"
        elif decision == "DISCARD":
            tag = "[DISCARD]"
        else:
            tag = "[FAIL]"

        lines = self.read_text().split("\n")
        settled_id: Optional[str] = None
        settled_desc = ""
        active_idx: Optional[int] = None

        # 1. Settle ACTIVE item.
        for i, line in enumerate(lines):
            m = _PLAN_ITEM_RE.match(line)
            if m is None or m.group(1) != " " or "(ACTIVE)" not in line:
                continue
            active_idx = i
            settled_id = m.group(2)
            rest = m.group(3).replace("(ACTIVE)", "").strip().lstrip(": ").strip()
            settled_desc = rest
            b = line.index("[ ]")
            lines[i] = line[:b] + f"[x] **{settled_id}** {tag}: {rest}"
            break

        if active_idx is None:
            raise RuntimeError("no (ACTIVE) item found in plan.md")

        # 2. Advance ACTIVE to next pending.
        for i, line in enumerate(lines):
            if i == active_idx:
                continue
            m = _PLAN_ITEM_RE.match(line)
            if m is None or m.group(1) != " " or "(ACTIVE)" in line:
                continue
            item_id = m.group(2)
            rest = m.group(3).lstrip(": ").strip()
            b = line.index("[ ]")
            lines[i] = line[:b] + f"[ ] **{item_id}** (ACTIVE): {rest}"
            break

        # 3. Append to Settled History table.
        history_line = (
            f"| {settled_id} | {decision} | "
            f"{metric if metric is not None else 'N/A'} | {settled_desc} |"
        )
        table_end: Optional[int] = None
        for i, line in enumerate(lines):
            if is_settled_table_header(line):
                table_end = i + 2  # skip header + separator
                for j in range(i + 2, len(lines)):
                    if lines[j].strip().startswith("|"):
                        table_end = j + 1
                    else:
                        break
        if table_end is not None:
            lines.insert(table_end, history_line)

        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return settled_id, settled_desc
