#!/usr/bin/env python3
"""
query_strategies.py — 策略卡片程序化筛选工具

读取 .claude/skills/evolution-strategies/references/cards/*.md 头部的 YAML frontmatter，
根据瓶颈类型、算子族、复杂度等条件筛选候选策略 ID。

用法:
    # 按瓶颈筛选
    python3 query_strategies.py --bottleneck mte2_stall --limit 15

    # 按算子族筛选
    python3 query_strategies.py --op-family attention

    # 复合筛选（AND）
    python3 query_strategies.py --bottleneck mte2_stall --op-family normalization \
        --complexity-max L1 --exclude-ids P1,P19 --output /tmp/candidates.json

    # 冲突排除
    python3 query_strategies.py --bottleneck mte2_stall --exclude-conflicts-of P14

    # 校验所有卡片 frontmatter
    python3 query_strategies.py --validate-all

输出 JSON:
    {
      "matched": ["P1", "P7", "P10"],
      "scored": [{"id": "P1", "score": 5, "hit_bottlenecks": [...], "hit_op_families": [...]}],
      "filter_applied": {...},
      "total_before_filter": 103,
      "total_after_filter": 3
    }

设计原则:
- 不引入 PyYAML 依赖（手写简化 YAML parser）
- 筛选结果带打分（命中瓶颈数×2 + 命中算子族数×1 + 协同命中×1）
- --exclude-conflicts-of 支持多个 ID（逗号分隔）
"""
import argparse
import glob
import json
import os
import re
import sys


VALID_BOTTLENECKS = {
    "mte2_stall", "mte3_stall", "tiling_imbalance", "scalar_loading",
    "scalar_compute", "compute_bound", "near_optimal", "no_overlap",
    "partial_overlap", "undersize_transfer", "icache_miss", "bus_contention",
    "l2_cache_thrash", "ub_memory_pressure",
}

VALID_OP_FAMILIES = {
    "elementwise", "normalization", "reduction", "softmax", "attention",
    "flash_attention", "cv_fusion", "matmul", "moe", "quantization",
    "pooling_gather", "optimizer", "index_scatter", "broadcast_mask",
    "special", "omni",
}

VALID_COMPLEXITIES = ["L0", "L1", "L2"]


def parse_frontmatter(content: str) -> dict | None:
    """从文件内容提取 YAML frontmatter（简化 parser，只支持本项目的字段格式）。

    支持：
        key: value
        key: [a, b, c]
        key:
          - a
          - b
    """
    if not content.startswith('---\n'):
        return None
    end_match = re.search(r'\n---\n', content[4:])
    if not end_match:
        return None
    yaml_block = content[4:4 + end_match.start()]

    result = {}
    current_list_key = None
    for line in yaml_block.split('\n'):
        # 跳过空行和注释
        if not line.strip() or line.strip().startswith('#'):
            continue

        # 列表项（以 - 开头，可缩进）
        if re.match(r'^\s+-\s', line):
            if current_list_key:
                item = line.strip()[2:].strip()
                # 去除可能的 quote
                item = item.strip('"').strip("'")
                if item:
                    result[current_list_key].append(item)
            continue

        # key: value 或 key: [list]
        m = re.match(r'^([\w_]+)\s*:\s*(.*)$', line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        current_list_key = None

        if not val:
            # key: （后续是 - 列表）
            result[key] = []
            current_list_key = key
        elif val.startswith('[') and val.endswith(']'):
            # key: [a, b, c]
            inner = val[1:-1].strip()
            if inner:
                items = [x.strip().strip('"').strip("'") for x in inner.split(',')]
                result[key] = [x for x in items if x]
            else:
                result[key] = []
        else:
            # key: value
            val = val.strip('"').strip("'")
            result[key] = val

    return result


def load_all_cards(cards_dir: str) -> dict:
    """扫描所有卡片，返回 {id: metadata_dict}。缺失 frontmatter 的卡片跳过。"""
    index = {}
    errors = []
    card_files = sorted(glob.glob(os.path.join(cards_dir, "*.md")))
    # 排除 SCHEMA.md / README.md 等辅助文件
    card_files = [f for f in card_files if not os.path.basename(f) in ("SCHEMA.md", "README.md")]

    for card_path in card_files:
        try:
            with open(card_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            errors.append({"file": card_path, "error": f"read_error: {e}"})
            continue

        fm = parse_frontmatter(content)
        if fm is None:
            # 没有 frontmatter，跳过（向后兼容，不算 error）
            continue

        sid = fm.get('id')
        if not sid:
            errors.append({"file": card_path, "error": "missing_id"})
            continue

        # 标准化列表字段
        fm['bottlenecks'] = fm.get('bottlenecks', []) or []
        fm['op_families'] = fm.get('op_families', []) or []
        fm['conflicts_with'] = fm.get('conflicts_with', []) or []
        fm['synergizes_with'] = fm.get('synergizes_with', []) or []
        fm['complexity'] = fm.get('complexity', 'L1')
        fm['_card_path'] = card_path

        index[sid] = fm

    return {"index": index, "errors": errors}


def validate_all(cards_dir: str) -> int:
    """校验所有卡片 frontmatter。返回错误数。"""
    data = load_all_cards(cards_dir)
    index = data["index"]
    errors = list(data["errors"])

    all_ids = set(index.keys())
    for sid, meta in index.items():
        # bottlenecks 合法性
        for b in meta['bottlenecks']:
            if b not in VALID_BOTTLENECKS:
                errors.append({"id": sid, "error": f"invalid_bottleneck: {b}"})
        # op_families 合法性
        for f in meta['op_families']:
            if f not in VALID_OP_FAMILIES:
                errors.append({"id": sid, "error": f"invalid_op_family: {f}"})
        # complexity 合法性
        if meta['complexity'] not in VALID_COMPLEXITIES:
            errors.append({"id": sid, "error": f"invalid_complexity: {meta['complexity']}"})
        # conflicts_with / synergizes_with 必须指向存在的 ID
        for cid in meta['conflicts_with']:
            if cid not in all_ids:
                errors.append({"id": sid, "error": f"conflicts_with_unknown_id: {cid}"})
        for sid2 in meta['synergizes_with']:
            if sid2 not in all_ids:
                errors.append({"id": sid, "error": f"synergizes_with_unknown_id: {sid2}"})
        # id 与文件名前缀一致性
        fname = os.path.basename(meta['_card_path'])
        prefix = fname.split('_', 1)[0].replace('.md', '')
        if prefix != sid:
            errors.append({"id": sid, "error": f"id_filename_mismatch: file={fname}"})

    print(f"Validation: {len(index)} cards loaded, {len(errors)} errors")
    for e in errors:
        print(f"  ❌ {e}")
    return len(errors)


def query(
    cards_dir: str,
    bottleneck: str = None,
    op_family: str = None,
    complexity_max: str = None,
    exclude_ids: list = None,
    exclude_conflicts_of: list = None,
    limit: int = 20,
) -> dict:
    """筛选策略。硬条件 AND 匹配，软条件参与打分。"""
    data = load_all_cards(cards_dir)
    index = data["index"]
    exclude_ids = set(exclude_ids or [])
    exclude_conflicts_of = exclude_conflicts_of or []

    # 展开 exclude_conflicts_of: 所有被这些 ID 冲突的策略都要排除
    for ref_id in exclude_conflicts_of:
        if ref_id in index:
            for conflict_id in index[ref_id].get('conflicts_with', []):
                exclude_ids.add(conflict_id)

    complexity_order = {"L0": 0, "L1": 1, "L2": 2}
    complexity_ceiling = complexity_order.get(complexity_max, 99) if complexity_max else 99

    candidates = []
    for sid, meta in index.items():
        # 排除 ID
        if sid in exclude_ids:
            continue

        # 复杂度上限
        if complexity_order.get(meta['complexity'], 99) > complexity_ceiling:
            continue

        # bottleneck 硬条件（如指定）
        hit_bottlenecks = []
        if bottleneck:
            if bottleneck not in meta['bottlenecks']:
                continue
            hit_bottlenecks.append(bottleneck)

        # op_family 硬条件（如指定）
        hit_op_families = []
        if op_family:
            if op_family in meta['op_families']:
                hit_op_families.append(op_family)
            elif 'omni' in meta['op_families']:
                # omni 视为通用命中（得分较低）
                hit_op_families.append('omni')
            else:
                continue

        # 打分：瓶颈×2 + 算子族×1 + 精确匹配 op_family 额外+1
        score = len(hit_bottlenecks) * 2 + len(hit_op_families)
        if op_family and op_family in meta['op_families']:
            score += 1  # 精确命中 op_family 额外加分（而非 omni 回退）

        candidates.append({
            "id": sid,
            "score": score,
            "complexity": meta['complexity'],
            "hit_bottlenecks": hit_bottlenecks,
            "hit_op_families": hit_op_families,
            "bottlenecks": meta['bottlenecks'],
            "op_families": meta['op_families'],
            "conflicts_with": meta['conflicts_with'],
            "synergizes_with": meta['synergizes_with'],
        })

    # 按得分降序排序，同分按 ID 升序
    def id_sort_key(sid):
        m = re.match(r'([A-Z]+)(\d+)', sid)
        if m:
            return (m.group(1), int(m.group(2)))
        return (sid, 0)

    candidates.sort(key=lambda c: (-c['score'], id_sort_key(c['id'])))
    candidates = candidates[:limit]

    return {
        "matched": [c['id'] for c in candidates],
        "scored": candidates,
        "filter_applied": {
            "bottleneck": bottleneck,
            "op_family": op_family,
            "complexity_max": complexity_max,
            "exclude_ids": sorted(exclude_ids),
        },
        "total_before_filter": len(index),
        "total_after_filter": len(candidates),
    }


def main():
    parser = argparse.ArgumentParser(description="策略卡片程序化筛选工具")
    parser.add_argument("--cards-dir", default=".claude/skills/evolution-strategies/references/cards/")
    parser.add_argument("--bottleneck", default=None, help="瓶颈类型（单值）")
    parser.add_argument("--op-family", default=None, help="算子族（单值）")
    parser.add_argument("--complexity-max", default=None, choices=VALID_COMPLEXITIES,
                        help="复杂度上限（L0/L1/L2）")
    parser.add_argument("--exclude-ids", default="", help="排除的策略 ID（逗号分隔）")
    parser.add_argument("--exclude-conflicts-of", default="",
                        help="排除与这些 ID 冲突的策略（逗号分隔）")
    parser.add_argument("--limit", type=int, default=20, help="返回候选数量上限")
    parser.add_argument("--output", default="-", help="输出 JSON 路径（- 表示 stdout）")
    parser.add_argument("--validate-all", action="store_true", help="校验所有卡片")

    args = parser.parse_args()

    # 参数校验
    if args.bottleneck and args.bottleneck not in VALID_BOTTLENECKS:
        print(f"ERROR: invalid bottleneck '{args.bottleneck}'. Valid: {sorted(VALID_BOTTLENECKS)}",
              file=sys.stderr)
        sys.exit(2)
    if args.op_family and args.op_family not in VALID_OP_FAMILIES:
        print(f"ERROR: invalid op_family '{args.op_family}'. Valid: {sorted(VALID_OP_FAMILIES)}",
              file=sys.stderr)
        sys.exit(2)

    if args.validate_all:
        err_count = validate_all(args.cards_dir)
        sys.exit(1 if err_count > 0 else 0)

    exclude_ids = [x.strip() for x in args.exclude_ids.split(',') if x.strip()]
    exclude_conflicts_of = [x.strip() for x in args.exclude_conflicts_of.split(',') if x.strip()]

    result = query(
        cards_dir=args.cards_dir,
        bottleneck=args.bottleneck,
        op_family=args.op_family,
        complexity_max=args.complexity_max,
        exclude_ids=exclude_ids,
        exclude_conflicts_of=exclude_conflicts_of,
        limit=args.limit,
    )

    out = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(out)
        print(f"Query result written: {args.output} ({result['total_after_filter']} matched)")


if __name__ == "__main__":
    main()
