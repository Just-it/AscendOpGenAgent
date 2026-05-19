#!/usr/bin/env python3
"""
precision_knowledge.py — 精度问题知识库管理

职责:
  1. load   — 全量加载知识库 (fallback 用)
  2. search — 结构化 RAG 检索: 按 op_type + pattern + position 评分排序, 返回 top-K + CHECKLIST
  3. dump   — 精度通过后, 将 Agent 生成的候选条目追加到知识库 (仅成功时调用)

知识库格式: 扁平五字段 JSON, 与现有 knowledge_base.json 结构对齐, RAG-ready。
每条记录:
  {
    "title":   "标准化中文标题 (含英文关键词)",
    "feature": "错误特征签名 (泛化, 中文)",
    "reason":  "深层原因 (中文)",
    "fix":     "通用修复指南 (代码级别, 中文)",
    "type":    "FIX_PRECISION_xxx"
  }

type 枚举 (精度专项):
  FIX_PRECISION_PADDING     — Padding 值导致精度问题
  FIX_PRECISION_TAIL        — 尾块处理精度问题
  FIX_PRECISION_REDUCTION   — 归约操作精度损失
  FIX_PRECISION_TYPECAST    — 类型转换精度问题
  FIX_PRECISION_LAYOUT      — 数据布局导致精度错误
  FIX_PRECISION_SYNC        — 同步问题导致精度随机错误
  FIX_PRECISION_OVERFLOW    — 数值溢出 (NaN/Inf)
  FIX_PRECISION_LOGIC       — 算法逻辑导致精度偏差
  FIX_PRECISION_OTHER       — 其他精度问题

用法:
    # 全量加载知识库 (fallback, stdout 输出 JSON)
    python3 precision_knowledge.py load --kb-path <path>

    # 结构化 RAG 检索 (推荐, stdout 输出 JSON)
    python3 precision_knowledge.py search --kb-path <path> --op-type <type> --pattern <hint> [--position <pos>] [--top-k 3]

    # 成功后写入知识库
    python3 precision_knowledge.py dump --kb-path <path> --output-path <path> --op-name <name>
"""

import argparse
import json
import os
import sys
from datetime import datetime


VALID_TYPES = [
    "FIX_PRECISION_PADDING",
    "FIX_PRECISION_TAIL",
    "FIX_PRECISION_REDUCTION",
    "FIX_PRECISION_TYPECAST",
    "FIX_PRECISION_LAYOUT",
    "FIX_PRECISION_SYNC",
    "FIX_PRECISION_OVERFLOW",
    "FIX_PRECISION_LOGIC",
    "FIX_PRECISION_OTHER",
]

REQUIRED_FIELDS = ["title", "feature", "reason", "fix", "type"]


# ============================================================
# Load
# ============================================================

def load_knowledge_base(kb_path: str) -> list:
    """加载知识库, 返回条目列表"""
    if not os.path.exists(kb_path):
        print(f"[KB] ⚠️ 知识库文件不存在: {kb_path}, 使用空知识库", file=sys.stderr)
        return []

    with open(kb_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[KB] ⚠️ 知识库格式错误 (期望 list), 使用空知识库", file=sys.stderr)
        return []

    # 过滤无效条目
    valid = []
    for entry in data:
        if all(entry.get(k) for k in REQUIRED_FIELDS):
            valid.append(entry)

    print(f"[KB] ✅ 已加载 {len(valid)} 条精度知识")
    print(json.dumps(valid, indent=2, ensure_ascii=False))
    return valid


# ============================================================
# Search (结构化 RAG 检索)
# ============================================================

# pattern → 最可能关联的 type 映射 (用于 type 字段交叉加权)
PATTERN_TYPE_AFFINITY = {
    "tail_spike": ["FIX_PRECISION_TAIL", "FIX_PRECISION_PADDING", "FIX_PRECISION_REDUCTION"],
    "uniform_offset": ["FIX_PRECISION_PADDING", "FIX_PRECISION_LOGIC", "FIX_PRECISION_REDUCTION"],
    "scattered": ["FIX_PRECISION_SYNC", "FIX_PRECISION_REDUCTION"],
    "magnitude_correlated": ["FIX_PRECISION_TYPECAST", "FIX_PRECISION_REDUCTION"],
    "nan_inf_contamination": ["FIX_PRECISION_OVERFLOW"],
    "dimension_concentration": ["FIX_PRECISION_LAYOUT", "FIX_PRECISION_LOGIC"],
    "boundary_concentration": ["FIX_PRECISION_PADDING", "FIX_PRECISION_TAIL"],
    "all_wrong": ["FIX_PRECISION_LOGIC", "FIX_PRECISION_LAYOUT", "FIX_PRECISION_REDUCTION", "FIX_PRECISION_SYNC"],
}

# position 特征 → 关联的 pattern (用于第二次检索的辅助加分)
POSITION_PATTERN_AFFINITY = {
    "tail": ["tail_spike", "boundary_concentration"],
    "boundary": ["boundary_concentration", "tail_spike"],
    "head": ["uniform_offset", "all_wrong"],
    "scattered": ["scattered", "magnitude_correlated"],
}

# 评分权重
W_PATTERN = 3
W_OP_TYPE = 2
W_TYPE = 1
W_POSITION = 1  # 第二次检索时 position 辅助加分
W_OP_TYPE_ALL_WRONG_BOOST = 2  # all_wrong 是泛化 hint, op_type 精确匹配时额外加权


def _is_checklist(entry: dict) -> bool:
    """判断条目是否为 CHECKLIST 类型"""
    return entry.get("title", "").startswith("[CHECKLIST]")


def _extract_patterns_from_feature(feature: str) -> list[str]:
    """从 feature 字段提取 pattern=xxx 标签"""
    patterns = []
    for token in feature.replace(",", " ").split():
        if token.startswith("pattern="):
            patterns.append(token.split("=", 1)[1])
    # 也检查 "或" 分隔的 pattern (如 "pattern=tail_spike 或 boundary_concentration")
    if "或" in feature:
        parts = feature.split("或")
        for part in parts:
            stripped = part.strip().rstrip(",").strip()
            # 如果 stripped 本身是一个 pattern 名 (无 = 前缀)
            if stripped in PATTERN_TYPE_AFFINITY and stripped not in patterns:
                patterns.append(stripped)
    return patterns


def _extract_op_type_from_feature(feature: str) -> str | None:
    """从 feature 字段提取 op_type=xxx 标签"""
    for token in feature.replace(",", " ").split():
        if token.startswith("op_type="):
            return token.split("=", 1)[1]
    return None


def _score_entry(entry: dict, query_pattern: str | None, query_op_type: str | None,
                 query_position: str | None) -> float:
    """对单条知识库条目评分"""
    score = 0.0
    feature = entry.get("feature", "")
    entry_type = entry.get("type", "")

    # 1. pattern 匹配 (权重 3)
    if query_pattern:
        entry_patterns = _extract_patterns_from_feature(feature)
        if query_pattern in entry_patterns:
            score += W_PATTERN
        # 部分匹配: feature 中包含 pattern 关键词但非标准 pattern=xxx 格式
        elif query_pattern in feature:
            score += W_PATTERN * 0.5

    # 2. op_type 匹配 (权重 2) — 仅对非 CHECKLIST 条目
    if query_op_type and not _is_checklist(entry):
        entry_op_type = _extract_op_type_from_feature(feature)
        if entry_op_type and entry_op_type == query_op_type:
            score += W_OP_TYPE
        # 名称推断: feature 中包含算子类型关键词
        op_type_keywords = {
            "reduction": ["归约", "reduce", "reduction"],
            "pooling": ["池化", "pool", "pooling"],
            "loss": ["损失", "loss"],
            "matmul": ["矩阵", "matmul", "gemm"],
            "normalization": ["归一化", "norm", "normalization"],
            "activation": ["激活", "activation", "exp", "sigmoid"],
        }
        if query_op_type in op_type_keywords:
            for kw in op_type_keywords[query_op_type]:
                if kw.lower() in feature.lower() or kw.lower() in entry.get("title", "").lower():
                    score += W_OP_TYPE * 0.5
                    break

    # 3. type 字段交叉匹配 (权重 1)
    if query_pattern and query_pattern in PATTERN_TYPE_AFFINITY:
        affine_types = PATTERN_TYPE_AFFINITY[query_pattern]
        if entry_type in affine_types:
            score += W_TYPE

    # 4. position 辅助加分 (仅第二次检索时使用, 权重 1)
    if query_position and query_position in POSITION_PATTERN_AFFINITY:
        affine_patterns = POSITION_PATTERN_AFFINITY[query_position]
        entry_patterns = _extract_patterns_from_feature(feature)
        for p in affine_patterns:
            if p in entry_patterns:
                score += W_POSITION
                break  # 只加一次

    # 5. all_wrong 特例: op_type 额外加权
    # all_wrong 是泛化 hint, 不携带具体问题信息, 此时 op_type 精确匹配是最重要的分类依据
    if query_pattern == "all_wrong" and query_op_type and not _is_checklist(entry):
        entry_op_type_check = _extract_op_type_from_feature(feature)
        if entry_op_type_check and entry_op_type_check == query_op_type:
            score += W_OP_TYPE_ALL_WRONG_BOOST

    return score


def search_knowledge_base(kb_path: str, op_type: str | None = None,
                          pattern: str | None = None, position: str | None = None,
                          top_k: int = 3) -> dict:
    """
    结构化 RAG 检索: 根据 op_type + pattern + position 筛选并评分排序。

    返回:
      {
        "query": {"op_type": ..., "pattern": ..., "position": ..., "top_k": ...},
        "matched_entries": [...],       # top-K 普通条目 (按 score 降序)
        "checklists": [...],            # op_type 匹配的 CHECKLIST (不占 K 配额)
        "total_kb_size": N,
        "fallback_to_full_load": bool   # 普通条目 + checklist 均为 0 时 True
      }
    """
    # 加载知识库
    if not os.path.exists(kb_path):
        print(f"[KB-SEARCH] ⚠️ 知识库文件不存在: {kb_path}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    try:
        with open(kb_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-SEARCH] ⚠️ 知识库读取失败: {e}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    if not isinstance(data, list):
        print(f"[KB-SEARCH] ⚠️ 知识库格式错误 (期望 list)", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    # 过滤无效条目
    valid = [e for e in data if all(e.get(k) for k in REQUIRED_FIELDS)]

    # 分离 CHECKLIST 和普通条目
    checklists = []
    normal_entries = []
    for entry in valid:
        if _is_checklist(entry):
            checklists.append(entry)
        else:
            normal_entries.append(entry)

    # CHECKLIST 按 op_type 精确匹配
    matched_checklists = []
    if op_type:
        for cl in checklists:
            cl_op_type = _extract_op_type_from_feature(cl.get("feature", ""))
            if cl_op_type and cl_op_type == op_type:
                matched_checklists.append(cl)

    # 普通条目评分排序
    scored = []
    for entry in normal_entries:
        score = _score_entry(entry, pattern, op_type, position)
        if score > 0:
            scored.append({"score": score, "entry": entry})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_entries = scored[:top_k]

    # Fallback: 无任何命中 → 全量 load
    fallback = len(top_entries) == 0 and len(matched_checklists) == 0
    if fallback:
        print(f"[KB-SEARCH] ⚠️ 无匹配条目, fallback 到全量加载", file=sys.stderr)
        # 全量返回所有条目
        top_entries = [{"score": 0, "entry": e} for e in normal_entries]
        matched_checklists = checklists

    result = {
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
        },
        "matched_entries": [
            {
                "index": valid.index(s["entry"]) if s["entry"] in valid else -1,
                "score": s["score"],
                "title": s["entry"]["title"],
                "feature": s["entry"]["feature"],
                "reason": s["entry"]["reason"],
                "fix": s["entry"]["fix"],
                "type": s["entry"]["type"],
            }
            for s in top_entries
        ],
        "checklists": [
            {
                "title": cl["title"],
                "feature": cl["feature"],
                "reason": cl["reason"],
                "fix": cl["fix"],
                "type": cl["type"],
            }
            for cl in matched_checklists
        ],
        "total_kb_size": len(valid),
        "fallback_to_full_load": fallback,
    }

    # 输出
    n_matched = len(result["matched_entries"])
    n_checklists = len(result["checklists"])
    print(f"[KB-SEARCH] ✅ 检索完成 (op_type={op_type}, pattern={pattern}, position={position})")
    print(f"  知识库总条目: {len(valid)}")
    print(f"  命中普通条目: {n_matched} / top-K={top_k}")
    print(f"  命中 CHECKLIST: {n_checklists}")
    if fallback:
        print(f"  ⚠️ FALLBACK: 无匹配, 已返回全量 {len(valid)} 条")
    for s in top_entries[:top_k]:
        print(f"    [{s['score']:.1f}] {s['entry']['title']}")
    for cl in matched_checklists:
        print(f"    [CL] {cl['title']}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _empty_search_result(op_type, pattern, position, top_k) -> dict:
    """空知识库时的返回结构"""
    return {
        "query": {"op_type": op_type, "pattern": pattern, "position": position, "top_k": top_k},
        "matched_entries": [],
        "checklists": [],
        "total_kb_size": 0,
        "fallback_to_full_load": True,
    }


# ============================================================
# Dump (仅精度通过时调用)
# ============================================================

def dump_success_knowledge(kb_path: str, output_path: str, op_name: str) -> dict | None:
    """
    从 Agent 生成的候选知识库条目追加到知识库。

    读取:
      - {output_path}/precision_tuning/candidate_kb_entry.json (Agent 生成的候选条目)
      - {output_path}/precision_tuning/forensics_report_{attempt}.json (用于补充元数据，使用最新一轮编号)

    逻辑:
      1. 读取候选条目 JSON，验证五字段完整性
      2. 验证 type 为合法枚举值
      3. 从取证报告补充元数据 (_meta)
      4. 按 title 去重
      5. 追加到知识库文件
    """
    tuning_dir = os.path.join(output_path, "precision_tuning")

    # 1. 读取 Agent 生成的候选条目
    candidate_path = os.path.join(tuning_dir, "candidate_kb_entry.json")
    if not os.path.exists(candidate_path):
        print(f"[KB] ⚠️ 候选条目文件不存在: {candidate_path}", file=sys.stderr)
        print(f"    请先执行 Step 5.1: 生成候选知识库条目", file=sys.stderr)
        return None

    try:
        with open(candidate_path) as f:
            candidate = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[KB] ⚠️ 候选条目 JSON 解析失败: {e}", file=sys.stderr)
        return None

    # 2. 验证五字段完整性
    missing = [f for f in REQUIRED_FIELDS if not candidate.get(f)]
    if missing:
        print(f"[KB] ⚠️ 候选条目缺少必填字段: {missing}", file=sys.stderr)
        return None

    # 3. 验证 type 枚举值
    if candidate["type"] not in VALID_TYPES:
        print(f"[KB] ⚠️ type 值非法: {candidate['type']}，应为以下之一: {VALID_TYPES}", file=sys.stderr)
        return None

    # 4. 补充元数据 (从最新取证报告读取，文件名格式 forensics_report_{attempt}.json)
    import glob as _glob
    _candidates = sorted(_glob.glob(os.path.join(tuning_dir, "forensics_report_*.json")))
    forensics_path = _candidates[-1] if _candidates else os.path.join(tuning_dir, "forensics_report.json")
    num_attempts = 1
    if os.path.exists(forensics_path):
        try:
            with open(forensics_path) as f:
                forensics = json.load(f)
            num_attempts = forensics.get("attempt", 0) + 1
            history_dir = os.path.join(tuning_dir, "history")
            if os.path.exists(history_dir):
                import re as _re
                attempt_dirs = [
                    d for d in os.listdir(history_dir)
                    if _re.match(r'^attempt_\d+$', d)
                    and os.path.isdir(os.path.join(history_dir, d))
                ]
                num_attempts = max(num_attempts, len(attempt_dirs))
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    entry = dict(candidate)  # 复制五字段
    entry["_meta"] = {
        "op_name": op_name,
        "created_at": datetime.now().isoformat(),
        "attempts_needed": num_attempts,
    }

    # 5. 去重 + 追加
    kb = []
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            kb = json.load(f)

    existing_titles = {e.get("title") for e in kb}
    if entry["title"] in existing_titles:
        print(f"[KB] ⚠️ 知识条目已存在 (title 重复), 跳过: {entry['title']}")
        return None

    kb.append(entry)

    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"[KB] ✅ 已写入知识条目: {entry['title']}")
    print(f"  type: {entry['type']}")
    print(f"  attempts_needed: {num_attempts}")
    print(f"  kb_size: {len(kb)} 条")
    return entry


# ============================================================
# CLI
# ============================================================

def _append_search_log(log_dir: str, call_index: int, op_type, pattern,
                       position, top_k: int, result: dict,
                       attempt: int | None = None) -> None:
    """
    追加写入知识库检索调用日志到 knowledge_search_log.json（统一文件名，含 attempt 字段）。
    若 log_dir 是目录，日志文件名固定为 knowledge_search_log.json；
    若 log_dir 以 .json 结尾则直接作为日志文件路径。
    """
    if log_dir.endswith(".json"):
        log_path = log_dir
    else:
        log_path = os.path.join(log_dir, "knowledge_search_log.json")

    existing_entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                existing_entries = json.load(f)
            if not isinstance(existing_entries, list):
                existing_entries = []
        except (json.JSONDecodeError, OSError):
            existing_entries = []

    entry = {
        "attempt": attempt,
        "call_index": call_index,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
        },
        "matched_count": len(result.get("matched_entries", [])),
        "checklist_count": len(result.get("checklists", [])),
        "fallback_to_full_load": result.get("fallback_to_full_load", False),
        "top_titles": [e["title"] for e in result.get("matched_entries", [])[:top_k]],
    }
    existing_entries.append(entry)

    try:
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing_entries, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="精度问题知识库管理")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # load (全量, fallback 用)
    p_load = subparsers.add_parser("load", help="全量加载知识库 (fallback)")
    p_load.add_argument("--kb-path", required=True, help="知识库 JSON 路径")

    # search (结构化 RAG 检索)
    p_search = subparsers.add_parser("search", help="结构化 RAG 检索")
    p_search.add_argument("--kb-path", required=True, help="知识库 JSON 路径")
    p_search.add_argument("--op-type", default=None,
                          help="算子类型 (来自 L8, 如 reduction/pooling/loss/matmul/normalization)")
    p_search.add_argument("--pattern", default=None,
                          help="误差模式 (来自取证 primary_hint, 如 tail_spike/uniform_offset)")
    p_search.add_argument("--position", default=None,
                          help="误差位置特征 (第二次检索用, 如 tail/boundary/head/scattered)")
    p_search.add_argument("--top-k", type=int, default=3,
                          help="返回普通条目数量上限 (默认 3, CHECKLIST 不占配额)")
    p_search.add_argument("--log-path", default=None,
                          help="知识库检索日志目录（如有，将追加写入 knowledge_search_log.json）")
    p_search.add_argument("--call-index", type=int, default=0,
                          help="本轮第几次检索（0-based），用于日志区分同轮多次调用")
    p_search.add_argument("--attempt", type=int, default=None,
                          help="当前调优轮次编号，写入日志 attempt 字段以区分多轮检索")

    # dump
    p_dump = subparsers.add_parser("dump", help="成功后写入知识库")
    p_dump.add_argument("--kb-path", required=True, help="知识库 JSON 路径")
    p_dump.add_argument("--output-path", required=True, help="算子输出目录")
    p_dump.add_argument("--op-name", required=True, help="算子名称")

    args = parser.parse_args()

    if args.command == "load":
        load_knowledge_base(args.kb_path)
    elif args.command == "search":
        result = search_knowledge_base(
            args.kb_path,
            op_type=args.op_type,
            pattern=args.pattern,
            position=args.position,
            top_k=args.top_k,
        )
        # 追加写入检索日志（可观测性：每次调用记录查询条件 + 命中数量）
        if getattr(args, "log_path", None):
            _append_search_log(
                log_dir=args.log_path,
                call_index=getattr(args, "call_index", 0),
                op_type=args.op_type,
                pattern=args.pattern,
                position=args.position,
                top_k=args.top_k,
                result=result,
                attempt=getattr(args, "attempt", None),
            )
    elif args.command == "dump":
        dump_success_knowledge(args.kb_path, args.output_path, args.op_name)


if __name__ == "__main__":
    main()
