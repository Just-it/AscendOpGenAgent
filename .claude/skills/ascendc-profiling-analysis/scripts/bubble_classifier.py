#!/usr/bin/env python3
"""
bubble_classifier.py — 空泡分类引擎 (两级分类体系)

一级分类 (BubbleCategory): 7 大类
二级分类 (BubbleSubType): ~30 个子类型

保留旧 BubbleType (A/B/C/D) 向后兼容。

用法:
    from bubble_classifier import classify_gap, BubbleType, BubbleCategory, BubbleSubType
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from trace_parser import (
    TraceEvent, DRAIN_OPS,
    PID_SCALAR, PID_SCALARLDST, PID_MTE2, PID_MTE3, PID_CACHEMISS, PID_FLOWCTRL,
    PID_MTE1, PID_CUBE, PID_FIXPIPE, PID_VECTOR,
    PIPELINE_NAME_MAP,
    compute_pipeline_coverage, has_cachemiss_in_window, has_flowctrl_in_window,
    is_cubecore,
)


# ── 旧四类枚举 (向后兼容) ──────────────────────────────────

class BubbleType(Enum):
    A = "a_class"  # 正常发射间距
    B = "b_class"  # Pipeline drain / barrier
    C = "c_class"  # 标量参数加载阻塞
    D = "d_class"  # 跨 pipeline 同步等待


# ── 新一级分类 ──────────────────────────────────────────────

class BubbleCategory(Enum):
    """一级分类 — 7 大类"""
    NORMAL = "normal"                     # 正常发射间距，不可优化
    STRUCTURAL = "structural"             # 流水线结构性开销 (drain/barrier)，不可优化
    DATA_STALL = "data_stall"             # 数据搬运等待 (MTE2/MTE3)，★★可优化
    SCALAR_OVERHEAD = "scalar_overhead"   # 标量计算/加载阻塞，★可优化
    RESOURCE_CONTENTION = "resource_contention"  # 资源竞争 (UB/icache)
    CROSS_CORE = "cross_core"             # 跨核不均衡
    CUBE_VECTOR = "cube_vector"           # Cube-Vector 协同 (MatMul 类)


# ── 新二级分类 ──────────────────────────────────────────────

class BubbleSubType(Enum):
    """二级分类 — ~30 个子类型"""
    # ── NORMAL ──
    N_ISSUE_GAP = "n_issue_gap"           # ≤1ps 正常发射间距

    # ── STRUCTURAL ──
    S_DRAIN = "s_drain"                   # drain 指令后的排空
    S_BARRIER = "s_barrier"               # PipeBarrier 同步
    S_COLD_START = "s_cold_start"         # 首个 tile 的流水线填充
    S_TAIL_DRAIN = "s_tail_drain"         # 最后一个 tile 的流水线排空
    S_ICACHE_MISS = "s_icache_miss"       # 指令缓存未命中 (全 idle + CACHEMISS)
    S_FLOWCTRL = "s_flowctrl"             # 流控指令开销

    # ── DATA_STALL ──
    D_MTE2_WAIT = "d_mte2_wait"           # VEC 等 MTE2 CopyIn (显式 WAIT_FLAG)
    D_MTE2_IMPLICIT = "d_mte2_implicit"   # VEC 等 MTE2 (隐式，无 WAIT_FLAG 但 MTE2 busy)
    D_MTE3_WAIT = "d_mte3_wait"           # VEC 等 MTE3 CopyOut (显式 WAIT_FLAG)
    D_MTE3_IMPLICIT = "d_mte3_implicit"   # VEC 等 MTE3 (隐式)
    D_MTE2_UNDERSIZE = "d_mte2_undersize" # MTE2 搬运粒度过小 (tile 太小)
    D_MTE3_UNDERSIZE = "d_mte3_undersize" # MTE3 搬运粒度过小
    D_NO_OVERLAP = "d_no_overlap"         # 缺少双缓冲导致无法重叠
    D_PARTIAL_OVERLAP = "d_partial_overlap"  # 有双缓冲但重叠不充分

    # ── SCALAR_OVERHEAD ──
    SC_LDST_BLOCK = "sc_ldst_block"       # SCALARLDST 参数加载阻塞 VEC
    SC_COMPUTE_BLOCK = "sc_compute_block" # SCALAR 地址计算阻塞 VEC
    SC_TILING_COMPLEX = "sc_tiling_complex"  # 复杂 tiling 逻辑导致标量开销大

    # ── RESOURCE_CONTENTION ──
    R_UB_PRESSURE = "r_ub_pressure"       # UB 内存不足导致频繁等待
    R_ICACHE_THRASH = "r_icache_thrash"   # 指令缓存抖动 (大内核)
    R_BUS_CONTENTION = "r_bus_contention" # 总线竞争 (MTE2+MTE3 同时搬运)

    # ── CROSS_CORE ──
    X_TILING_IMBALANCE = "x_tiling_imbalance"   # 分块不均导致核间负载差异
    X_TAIL_CORE = "x_tail_core"                 # 尾核工作量不足
    X_SYNC_BARRIER = "x_sync_barrier"           # 核间同步屏障等待

    # ── CUBE_VECTOR ──
    CV_CUBE_WAIT = "cv_cube_wait"         # Vector 等 Cube 完成
    CV_VECTOR_WAIT = "cv_vector_wait"     # Cube 等 Vector 完成
    CV_HANDOFF = "cv_handoff"             # Cube→Vector 数据交接开销


# ── 向后兼容映射 ────────────────────────────────────────────

SUBTYPE_TO_LEGACY: dict[BubbleSubType, BubbleType] = {
    BubbleSubType.N_ISSUE_GAP: BubbleType.A,
    BubbleSubType.S_DRAIN: BubbleType.B,
    BubbleSubType.S_BARRIER: BubbleType.B,
    BubbleSubType.S_COLD_START: BubbleType.B,
    BubbleSubType.S_TAIL_DRAIN: BubbleType.B,
    BubbleSubType.S_ICACHE_MISS: BubbleType.B,
    BubbleSubType.S_FLOWCTRL: BubbleType.B,
    BubbleSubType.SC_LDST_BLOCK: BubbleType.C,
    BubbleSubType.SC_COMPUTE_BLOCK: BubbleType.C,
    BubbleSubType.SC_TILING_COMPLEX: BubbleType.C,
    BubbleSubType.D_MTE2_WAIT: BubbleType.D,
    BubbleSubType.D_MTE2_IMPLICIT: BubbleType.D,
    BubbleSubType.D_MTE3_WAIT: BubbleType.D,
    BubbleSubType.D_MTE3_IMPLICIT: BubbleType.D,
    BubbleSubType.D_MTE2_UNDERSIZE: BubbleType.D,
    BubbleSubType.D_MTE3_UNDERSIZE: BubbleType.D,
    BubbleSubType.D_NO_OVERLAP: BubbleType.D,
    BubbleSubType.D_PARTIAL_OVERLAP: BubbleType.D,
    # RESOURCE_CONTENTION → B (不可直接优化)
    BubbleSubType.R_UB_PRESSURE: BubbleType.B,
    BubbleSubType.R_ICACHE_THRASH: BubbleType.B,
    BubbleSubType.R_BUS_CONTENTION: BubbleType.B,
    # CROSS_CORE → D
    BubbleSubType.X_TILING_IMBALANCE: BubbleType.D,
    BubbleSubType.X_TAIL_CORE: BubbleType.D,
    BubbleSubType.X_SYNC_BARRIER: BubbleType.D,
    # CUBE_VECTOR → D
    BubbleSubType.CV_CUBE_WAIT: BubbleType.D,
    BubbleSubType.CV_VECTOR_WAIT: BubbleType.D,
    BubbleSubType.CV_HANDOFF: BubbleType.D,
}

SUBTYPE_TO_CATEGORY: dict[BubbleSubType, BubbleCategory] = {
    BubbleSubType.N_ISSUE_GAP: BubbleCategory.NORMAL,
    BubbleSubType.S_DRAIN: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_BARRIER: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_COLD_START: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_TAIL_DRAIN: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_ICACHE_MISS: BubbleCategory.STRUCTURAL,
    BubbleSubType.S_FLOWCTRL: BubbleCategory.STRUCTURAL,
    BubbleSubType.SC_LDST_BLOCK: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.SC_COMPUTE_BLOCK: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.SC_TILING_COMPLEX: BubbleCategory.SCALAR_OVERHEAD,
    BubbleSubType.D_MTE2_WAIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE2_IMPLICIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_WAIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_IMPLICIT: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE2_UNDERSIZE: BubbleCategory.DATA_STALL,
    BubbleSubType.D_MTE3_UNDERSIZE: BubbleCategory.DATA_STALL,
    BubbleSubType.D_NO_OVERLAP: BubbleCategory.DATA_STALL,
    BubbleSubType.D_PARTIAL_OVERLAP: BubbleCategory.DATA_STALL,
    BubbleSubType.R_UB_PRESSURE: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.R_ICACHE_THRASH: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.R_BUS_CONTENTION: BubbleCategory.RESOURCE_CONTENTION,
    BubbleSubType.X_TILING_IMBALANCE: BubbleCategory.CROSS_CORE,
    BubbleSubType.X_TAIL_CORE: BubbleCategory.CROSS_CORE,
    BubbleSubType.X_SYNC_BARRIER: BubbleCategory.CROSS_CORE,
    BubbleSubType.CV_CUBE_WAIT: BubbleCategory.CUBE_VECTOR,
    BubbleSubType.CV_VECTOR_WAIT: BubbleCategory.CUBE_VECTOR,
    BubbleSubType.CV_HANDOFF: BubbleCategory.CUBE_VECTOR,
}


# ── 量化 ConcurrentState ───────────────────────────────────

@dataclass
class ConcurrentState:
    """时间窗口内各 pipeline 的量化状态"""
    # 共享 pipeline
    scalar_coverage: float = 0.0       # 0.0~1.0
    scalarldst_coverage: float = 0.0
    mte2_coverage: float = 0.0
    # veccore 专有
    mte3_coverage: float = 0.0
    # cubecore 专有
    mte1_coverage: float = 0.0
    cube_coverage: float = 0.0
    fixpipe_coverage: float = 0.0
    # 标记
    cachemiss_active: bool = False
    flowctrl_active: bool = False
    # ops 列表
    scalar_ops: list = None
    scalarldst_ops: list = None
    mte2_ops: list = None
    mte3_ops: list = None
    mte1_ops: list = None
    cube_ops: list = None
    fixpipe_ops: list = None

    def __post_init__(self):
        if self.scalar_ops is None:
            self.scalar_ops = []
        if self.scalarldst_ops is None:
            self.scalarldst_ops = []
        if self.mte2_ops is None:
            self.mte2_ops = []
        if self.mte3_ops is None:
            self.mte3_ops = []
        if self.mte1_ops is None:
            self.mte1_ops = []
        if self.cube_ops is None:
            self.cube_ops = []
        if self.fixpipe_ops is None:
            self.fixpipe_ops = []

    # 向后兼容属性
    @property
    def scalar_busy(self) -> bool:
        return self.scalar_coverage > 0.05

    @property
    def scalarldst_busy(self) -> bool:
        return self.scalarldst_coverage > 0.05

    @property
    def mte2_busy(self) -> bool:
        return self.mte2_coverage > 0.05

    @property
    def mte3_busy(self) -> bool:
        return self.mte3_coverage > 0.05

    @property
    def mte1_busy(self) -> bool:
        return self.mte1_coverage > 0.05

    @property
    def cube_busy(self) -> bool:
        return self.cube_coverage > 0.05

    @property
    def fixpipe_busy(self) -> bool:
        return self.fixpipe_coverage > 0.05

    @property
    def all_idle(self) -> bool:
        return not (self.scalar_busy or self.scalarldst_busy
                    or self.mte2_busy or self.mte3_busy
                    or self.mte1_busy or self.cube_busy
                    or self.fixpipe_busy)


# ── BubbleClassification ───────────────────────────────────

@dataclass
class BubbleClassification:
    """空泡分类结果"""
    bubble_type: BubbleType           # 旧四类 (向后兼容)
    category: BubbleCategory          # 新一级分类
    sub_type: BubbleSubType           # 新二级分类
    gap_ps: float
    reason: str
    optimizable: bool
    optimization_hint: Optional[str] = None
    waited_pipeline: Optional[str] = None
    concurrent_cause: Optional[str] = None
    coverage_detail: Optional[dict] = None  # 各 pipeline 覆盖率快照


# ── 阈值常量 ──────────────────────────────────────────────

THRESHOLD_A_MAX_PS = 1.0       # ≤1ps → A 类
THRESHOLD_B_DRAIN_PS = 30.0    # 1-30ps drain → B 类
THRESHOLD_C_MAX_PS = 500.0     # 30-500ps → C 类候选
# >500ps → D 类候选

# barrier 指令名
BARRIER_OPS = frozenset({"BAR", "PipeBarrier", "PIPE_BARRIER"})


def _make_classification(
    sub_type: BubbleSubType,
    gap_ps: float,
    reason: str,
    optimizable: bool,
    optimization_hint: Optional[str] = None,
    waited_pipeline: Optional[str] = None,
    concurrent_cause: Optional[str] = None,
    coverage_detail: Optional[dict] = None,
) -> BubbleClassification:
    """从 sub_type 自动推导 bubble_type 和 category"""
    return BubbleClassification(
        bubble_type=SUBTYPE_TO_LEGACY.get(sub_type, BubbleType.B),
        category=SUBTYPE_TO_CATEGORY.get(sub_type, BubbleCategory.STRUCTURAL),
        sub_type=sub_type,
        gap_ps=gap_ps,
        reason=reason,
        optimizable=optimizable,
        optimization_hint=optimization_hint,
        waited_pipeline=waited_pipeline,
        concurrent_cause=concurrent_cause,
        coverage_detail=coverage_detail,
    )


def classify_gap(
    gap_ps: float,
    op_before: Optional[TraceEvent],
    op_after: Optional[TraceEvent],
    concurrent_state: Optional[ConcurrentState] = None,
    has_wait_flag: bool = False,
    wait_flag_event: Optional[TraceEvent] = None,
    # ── 新增上下文参数 ──
    iteration_index: Optional[int] = None,
    total_iterations: Optional[int] = None,
    core_type: str = "veccore",
) -> BubbleClassification:
    """
    对 pipeline 中两条相邻指令之间的间隙进行空泡分类。

    Args:
        gap_ps: 间隙时长 (ps)
        op_before: 间隙前的事件 (可为 None)
        op_after: 间隙后的事件 (可为 None)
        concurrent_state: 间隙期间其他 pipeline 的状态
        has_wait_flag: 间隙中是否包含 WAIT_FLAG 事件
        wait_flag_event: WAIT_FLAG 事件本身 (用于提取被等待 pipeline)
        iteration_index: 当前 tile 序号 (用于冷启动/尾部判定)
        total_iterations: 总 tile 数
        core_type: "veccore" 或 "cubecore"

    Returns:
        BubbleClassification
    """
    if core_type == "cubecore":
        return _classify_gap_cube(
            gap_ps, op_before, op_after, concurrent_state,
            has_wait_flag, wait_flag_event,
            iteration_index, total_iterations,
        )
    return _classify_gap_vec(
        gap_ps, op_before, op_after, concurrent_state,
        has_wait_flag, wait_flag_event,
        iteration_index, total_iterations,
    )


def _classify_gap_vec(
    gap_ps: float,
    op_before: Optional[TraceEvent],
    op_after: Optional[TraceEvent],
    concurrent_state: Optional[ConcurrentState] = None,
    has_wait_flag: bool = False,
    wait_flag_event: Optional[TraceEvent] = None,
    iteration_index: Optional[int] = None,
    total_iterations: Optional[int] = None,
) -> BubbleClassification:
    """
    对 VEC pipeline 中两条相邻指令之间的间隙进行空泡分类 (veccore)。
    """
    if concurrent_state is None:
        concurrent_state = ConcurrentState()

    coverage = {
        "scalar": concurrent_state.scalar_coverage,
        "scalarldst": concurrent_state.scalarldst_coverage,
        "mte2": concurrent_state.mte2_coverage,
        "mte3": concurrent_state.mte3_coverage,
        "cachemiss": concurrent_state.cachemiss_active,
        "flowctrl": concurrent_state.flowctrl_active,
    }

    # ── 规则 1: 极小间隙 → N_ISSUE_GAP ──
    if gap_ps <= THRESHOLD_A_MAX_PS:
        return _make_classification(
            BubbleSubType.N_ISSUE_GAP, gap_ps,
            "Normal issue gap", False,
            coverage_detail=coverage,
        )

    # ── 规则 2: 小间隙 (1-30ps) ──
    if gap_ps <= THRESHOLD_B_DRAIN_PS:
        if op_before and op_before.is_drain_op():
            if op_before.name in BARRIER_OPS:
                return _make_classification(
                    BubbleSubType.S_BARRIER, gap_ps,
                    f"PipeBarrier sync after {op_before.name}", False,
                    coverage_detail=coverage,
                )
            return _make_classification(
                BubbleSubType.S_DRAIN, gap_ps,
                f"Pipeline drain after {op_before.name}", False,
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.N_ISSUE_GAP, gap_ps,
            "Small gap, non-drain", False,
            coverage_detail=coverage,
        )

    # ── 规则 3: 中等间隙 (30-500ps) ──
    if gap_ps <= THRESHOLD_C_MAX_PS:
        if concurrent_state.cachemiss_active:
            return _make_classification(
                BubbleSubType.S_ICACHE_MISS, gap_ps,
                "Instruction cache miss detected (CACHEMISS active)", False,
                coverage_detail=coverage,
            )
        if concurrent_state.flowctrl_active:
            return _make_classification(
                BubbleSubType.S_FLOWCTRL, gap_ps,
                "Flow control instruction overhead", False,
                coverage_detail=coverage,
            )
        if concurrent_state.scalarldst_coverage > 0.3:
            top_op = (concurrent_state.scalarldst_ops[0]
                      if concurrent_state.scalarldst_ops else "unknown")
            op_name = top_op.name if isinstance(top_op, TraceEvent) else str(top_op)
            return _make_classification(
                BubbleSubType.SC_LDST_BLOCK, gap_ps,
                f"SCALARLDST busy ({op_name}, coverage={concurrent_state.scalarldst_coverage:.0%})",
                True,
                optimization_hint="Parameter prefetch: preload next tile params during compute",
                concurrent_cause="SCALARLDST",
                coverage_detail=coverage,
            )
        if concurrent_state.scalar_coverage > 0.3:
            top_op = (concurrent_state.scalar_ops[0]
                      if concurrent_state.scalar_ops else "unknown")
            op_name = top_op.name if isinstance(top_op, TraceEvent) else str(top_op)
            return _make_classification(
                BubbleSubType.SC_COMPUTE_BLOCK, gap_ps,
                f"SCALAR busy ({op_name}, coverage={concurrent_state.scalar_coverage:.0%})",
                True,
                optimization_hint="Reduce address computation: simplify tiling or precompute offsets",
                concurrent_cause="SCALAR",
                coverage_detail=coverage,
            )
        if concurrent_state.all_idle:
            return _make_classification(
                BubbleSubType.S_DRAIN, gap_ps,
                "Extended drain / all pipelines idle", False,
                coverage_detail=coverage,
            )
        # MTE2/MTE3 busy in mid-range → data stall
        if concurrent_state.mte2_coverage > 0.3:
            return _make_classification(
                BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
                f"MTE2 busy in mid-range gap (coverage={concurrent_state.mte2_coverage:.0%})",
                True,
                optimization_hint="Enable double buffering (P1) to overlap MTE2 with VEC",
                waited_pipeline="MTE2",
                coverage_detail=coverage,
            )
        if concurrent_state.mte3_coverage > 0.3:
            return _make_classification(
                BubbleSubType.D_MTE3_IMPLICIT, gap_ps,
                f"MTE3 busy in mid-range gap (coverage={concurrent_state.mte3_coverage:.0%})",
                True,
                optimization_hint="Enable double buffering (P1) to overlap MTE3 with VEC",
                waited_pipeline="MTE3",
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.S_DRAIN, gap_ps,
            "Mid-range gap, non-scalar cause", False,
            coverage_detail=coverage,
        )

    # ── 规则 4: 大间隙 (>500ps) ──

    # 冷启动 / 尾部排空
    if iteration_index is not None and total_iterations is not None:
        if iteration_index == 0 and total_iterations > 1:
            return _make_classification(
                BubbleSubType.S_COLD_START, gap_ps,
                "First tile pipeline fill (cold start)", False,
                coverage_detail=coverage,
            )
        if iteration_index == total_iterations - 1 and total_iterations > 1:
            return _make_classification(
                BubbleSubType.S_TAIL_DRAIN, gap_ps,
                "Last tile pipeline drain (tail)", False,
                coverage_detail=coverage,
            )

    # 显式 WAIT_FLAG
    if has_wait_flag and wait_flag_event:
        waited = wait_flag_event.waited_pipeline()
        if waited == "MTE2" or (waited is None and concurrent_state.mte2_busy):
            return _make_classification(
                BubbleSubType.D_MTE2_WAIT, gap_ps,
                "WAIT_FLAG for MTE2 (data load stall)",
                True,
                optimization_hint="Enable double buffering (P1) or increase tile size to overlap MTE2 with VEC",
                waited_pipeline="MTE2",
                coverage_detail=coverage,
            )
        if waited == "MTE3" or (waited is None and concurrent_state.mte3_busy):
            return _make_classification(
                BubbleSubType.D_MTE3_WAIT, gap_ps,
                "WAIT_FLAG for MTE3 (write-back stall)",
                True,
                optimization_hint="Enable double buffering (P1) to overlap MTE3 write-back with next compute",
                waited_pipeline="MTE3",
                coverage_detail=coverage,
            )
        if waited == "SCALAR":
            return _make_classification(
                BubbleSubType.SC_COMPUTE_BLOCK, gap_ps,
                "WAIT_FLAG for SCALAR",
                True,
                optimization_hint="Reduce scalar dependency: simplify control flow or precompute",
                waited_pipeline="SCALAR",
                concurrent_cause="SCALAR",
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.D_MTE2_WAIT, gap_ps,
            f"WAIT_FLAG for {waited or 'unknown'}",
            True,
            optimization_hint="Investigate pipeline dependency",
            waited_pipeline=waited,
            coverage_detail=coverage,
        )

    if has_wait_flag:
        return _make_classification(
            BubbleSubType.D_MTE2_WAIT, gap_ps,
            "WAIT_FLAG (detail unavailable)",
            True,
            optimization_hint="Enable double buffering to overlap data transfer with compute",
            coverage_detail=coverage,
        )

    # 无 WAIT_FLAG — 隐式同步
    if concurrent_state.mte2_coverage > 0.5:
        return _make_classification(
            BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
            f"Implicit MTE2 stall (coverage={concurrent_state.mte2_coverage:.0%})",
            True,
            optimization_hint="Enable double buffering (P1) to overlap MTE2 with VEC",
            waited_pipeline="MTE2",
            coverage_detail=coverage,
        )
    if concurrent_state.mte3_coverage > 0.5:
        return _make_classification(
            BubbleSubType.D_MTE3_IMPLICIT, gap_ps,
            f"Implicit MTE3 stall (coverage={concurrent_state.mte3_coverage:.0%})",
            True,
            optimization_hint="Enable double buffering (P1) to overlap MTE3 with VEC",
            waited_pipeline="MTE3",
            coverage_detail=coverage,
        )
    if concurrent_state.all_idle and concurrent_state.cachemiss_active:
        return _make_classification(
            BubbleSubType.S_ICACHE_MISS, gap_ps,
            "All idle + CACHEMISS active → instruction cache miss", False,
            coverage_detail=coverage,
        )

    # 默认: 无法重叠
    return _make_classification(
        BubbleSubType.D_NO_OVERLAP, gap_ps,
        "Large gap without explicit WAIT_FLAG (implicit sync, no overlap)",
        True,
        optimization_hint="Investigate with T4 concurrent_pipeline_view for root cause",
        coverage_detail=coverage,
    )


def _classify_gap_cube(
    gap_ps: float,
    op_before: Optional[TraceEvent],
    op_after: Optional[TraceEvent],
    concurrent_state: Optional[ConcurrentState] = None,
    has_wait_flag: bool = False,
    wait_flag_event: Optional[TraceEvent] = None,
    iteration_index: Optional[int] = None,
    total_iterations: Optional[int] = None,
) -> BubbleClassification:
    """
    对 CUBE pipeline 中两条相邻指令之间的间隙进行空泡分类 (cubecore)。

    Cubecore pipeline:
      CUBE (MatMul) → FIXPIPE (L0C→UB) → MTE1 (L1→L0A/L0B) → MTE2 (GM→L1) → SCALAR/SCALARLDST
    """
    if concurrent_state is None:
        concurrent_state = ConcurrentState()

    coverage = {
        "scalar": concurrent_state.scalar_coverage,
        "scalarldst": concurrent_state.scalarldst_coverage,
        "mte1": concurrent_state.mte1_coverage,
        "cube": concurrent_state.cube_coverage,
        "mte2": concurrent_state.mte2_coverage,
        "fixpipe": concurrent_state.fixpipe_coverage,
        "cachemiss": concurrent_state.cachemiss_active,
        "flowctrl": concurrent_state.flowctrl_active,
    }

    # ── 规则 1: 极小间隙 → N_ISSUE_GAP ──
    if gap_ps <= THRESHOLD_A_MAX_PS:
        return _make_classification(
            BubbleSubType.N_ISSUE_GAP, gap_ps,
            "Normal issue gap", False,
            coverage_detail=coverage,
        )

    # ── 规则 2: 小间隙 (1-30ps) ──
    if gap_ps <= THRESHOLD_B_DRAIN_PS:
        if op_before and op_before.is_drain_op():
            if op_before.name in BARRIER_OPS:
                return _make_classification(
                    BubbleSubType.S_BARRIER, gap_ps,
                    f"PipeBarrier sync after {op_before.name}", False,
                    coverage_detail=coverage,
                )
            return _make_classification(
                BubbleSubType.S_DRAIN, gap_ps,
                f"Pipeline drain after {op_before.name}", False,
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.N_ISSUE_GAP, gap_ps,
            "Small gap, non-drain", False,
            coverage_detail=coverage,
        )

    # ── 规则 3: 中等间隙 (30-500ps) ──
    if gap_ps <= THRESHOLD_C_MAX_PS:
        if concurrent_state.cachemiss_active:
            return _make_classification(
                BubbleSubType.S_ICACHE_MISS, gap_ps,
                "Instruction cache miss detected (CACHEMISS active)", False,
                coverage_detail=coverage,
            )
        if concurrent_state.flowctrl_active:
            return _make_classification(
                BubbleSubType.S_FLOWCTRL, gap_ps,
                "Flow control instruction overhead", False,
                coverage_detail=coverage,
            )
        # Cubecore: FIXPIPE busy → L0C→UB conversion overhead
        if concurrent_state.fixpipe_coverage > 0.3:
            return _make_classification(
                BubbleSubType.CV_HANDOFF, gap_ps,
                f"FIXPIPE busy (coverage={concurrent_state.fixpipe_coverage:.0%}), L0C→UB conversion",
                False,
                optimization_hint="FIXPIPE overhead is structural (L0C→UB format conversion)",
                concurrent_cause="FIXPIPE",
                coverage_detail=coverage,
            )
        if concurrent_state.scalarldst_coverage > 0.3:
            top_op = (concurrent_state.scalarldst_ops[0]
                      if concurrent_state.scalarldst_ops else "unknown")
            op_name = top_op.name if isinstance(top_op, TraceEvent) else str(top_op)
            return _make_classification(
                BubbleSubType.SC_LDST_BLOCK, gap_ps,
                f"SCALARLDST busy ({op_name}, coverage={concurrent_state.scalarldst_coverage:.0%})",
                True,
                optimization_hint="Parameter prefetch: preload next tile params during compute",
                concurrent_cause="SCALARLDST",
                coverage_detail=coverage,
            )
        if concurrent_state.scalar_coverage > 0.3:
            top_op = (concurrent_state.scalar_ops[0]
                      if concurrent_state.scalar_ops else "unknown")
            op_name = top_op.name if isinstance(top_op, TraceEvent) else str(top_op)
            return _make_classification(
                BubbleSubType.SC_COMPUTE_BLOCK, gap_ps,
                f"SCALAR busy ({op_name}, coverage={concurrent_state.scalar_coverage:.0%})",
                True,
                optimization_hint="Reduce address computation: simplify tiling or precompute offsets",
                concurrent_cause="SCALAR",
                coverage_detail=coverage,
            )
        if concurrent_state.all_idle:
            return _make_classification(
                BubbleSubType.S_DRAIN, gap_ps,
                "Extended drain / all pipelines idle", False,
                coverage_detail=coverage,
            )
        # MTE1/MTE2 busy in mid-range → data stall
        if concurrent_state.mte1_coverage > 0.3:
            return _make_classification(
                BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
                f"MTE1 busy (L1→L0 load, coverage={concurrent_state.mte1_coverage:.0%})",
                True,
                optimization_hint="Overlap MTE1 data load with CUBE compute via double buffering",
                waited_pipeline="MTE1",
                coverage_detail=coverage,
            )
        if concurrent_state.mte2_coverage > 0.3:
            return _make_classification(
                BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
                f"MTE2 busy (GM→L1 load, coverage={concurrent_state.mte2_coverage:.0%})",
                True,
                optimization_hint="Enable double buffering to overlap MTE2 GM→L1 with compute",
                waited_pipeline="MTE2",
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.S_DRAIN, gap_ps,
            "Mid-range gap, non-scalar cause", False,
            coverage_detail=coverage,
        )

    # ── 规则 4: 大间隙 (>500ps) ──

    # 冷启动 / 尾部排空
    if iteration_index is not None and total_iterations is not None:
        if iteration_index == 0 and total_iterations > 1:
            return _make_classification(
                BubbleSubType.S_COLD_START, gap_ps,
                "First tile pipeline fill (cold start)", False,
                coverage_detail=coverage,
            )
        if iteration_index == total_iterations - 1 and total_iterations > 1:
            return _make_classification(
                BubbleSubType.S_TAIL_DRAIN, gap_ps,
                "Last tile pipeline drain (tail)", False,
                coverage_detail=coverage,
            )

    # 显式 WAIT_FLAG
    if has_wait_flag and wait_flag_event:
        waited = wait_flag_event.waited_pipeline()
        if waited == "MTE1" or (waited is None and concurrent_state.mte1_busy):
            return _make_classification(
                BubbleSubType.D_MTE2_WAIT, gap_ps,
                "WAIT_FLAG for MTE1 (L1→L0 data load stall)",
                True,
                optimization_hint="Enable double buffering to overlap MTE1 L1→L0 with CUBE compute",
                waited_pipeline="MTE1",
                coverage_detail=coverage,
            )
        if waited == "MTE2" or (waited is None and concurrent_state.mte2_busy):
            return _make_classification(
                BubbleSubType.D_MTE2_WAIT, gap_ps,
                "WAIT_FLAG for MTE2 (GM→L1 data load stall)",
                True,
                optimization_hint="Enable double buffering to overlap MTE2 GM→L1 with compute",
                waited_pipeline="MTE2",
                coverage_detail=coverage,
            )
        if waited == "FIXPIPE" or (waited is None and concurrent_state.fixpipe_busy):
            return _make_classification(
                BubbleSubType.CV_HANDOFF, gap_ps,
                "WAIT_FLAG for FIXPIPE (L0C→UB conversion)",
                False,
                optimization_hint="FIXPIPE overhead is structural (L0C→UB format conversion)",
                waited_pipeline="FIXPIPE",
                coverage_detail=coverage,
            )
        if waited == "SCALAR":
            return _make_classification(
                BubbleSubType.SC_COMPUTE_BLOCK, gap_ps,
                "WAIT_FLAG for SCALAR",
                True,
                optimization_hint="Reduce scalar dependency: simplify control flow or precompute",
                waited_pipeline="SCALAR",
                concurrent_cause="SCALAR",
                coverage_detail=coverage,
            )
        return _make_classification(
            BubbleSubType.D_MTE2_WAIT, gap_ps,
            f"WAIT_FLAG for {waited or 'unknown'}",
            True,
            optimization_hint="Investigate pipeline dependency",
            waited_pipeline=waited,
            coverage_detail=coverage,
        )

    if has_wait_flag:
        return _make_classification(
            BubbleSubType.D_MTE2_WAIT, gap_ps,
            "WAIT_FLAG (detail unavailable)",
            True,
            optimization_hint="Enable double buffering to overlap data transfer with compute",
            coverage_detail=coverage,
        )

    # 无 WAIT_FLAG — 隐式同步
    if concurrent_state.mte1_coverage > 0.5:
        return _make_classification(
            BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
            f"Implicit MTE1 stall (L1→L0, coverage={concurrent_state.mte1_coverage:.0%})",
            True,
            optimization_hint="Enable double buffering to overlap MTE1 with CUBE",
            waited_pipeline="MTE1",
            coverage_detail=coverage,
        )
    if concurrent_state.mte2_coverage > 0.5:
        return _make_classification(
            BubbleSubType.D_MTE2_IMPLICIT, gap_ps,
            f"Implicit MTE2 stall (GM→L1, coverage={concurrent_state.mte2_coverage:.0%})",
            True,
            optimization_hint="Enable double buffering to overlap MTE2 with CUBE",
            waited_pipeline="MTE2",
            coverage_detail=coverage,
        )
    if concurrent_state.fixpipe_coverage > 0.5:
        return _make_classification(
            BubbleSubType.CV_HANDOFF, gap_ps,
            f"Implicit FIXPIPE stall (coverage={concurrent_state.fixpipe_coverage:.0%})",
            False,
            optimization_hint="FIXPIPE overhead is structural (L0C→UB format conversion)",
            waited_pipeline="FIXPIPE",
            coverage_detail=coverage,
        )
    if concurrent_state.all_idle and concurrent_state.cachemiss_active:
        return _make_classification(
            BubbleSubType.S_ICACHE_MISS, gap_ps,
            "All idle + CACHEMISS active → instruction cache miss", False,
            coverage_detail=coverage,
        )

    # 默认: 无法重叠
    return _make_classification(
        BubbleSubType.D_NO_OVERLAP, gap_ps,
        "Large gap without explicit WAIT_FLAG (implicit sync, no overlap)",
        True,
        optimization_hint="Investigate with T4 concurrent_pipeline_view for root cause",
        coverage_detail=coverage,
    )


def build_concurrent_state(
    concurrent_events: dict[int, list[TraceEvent]],
    pipelines: dict[int, list[TraceEvent]] = None,
    window_start: float = 0,
    window_end: float = 0,
    core_type: str = "veccore",
) -> ConcurrentState:
    """
    从 find_concurrent_events() 的输出构建 ConcurrentState。

    如果提供了 pipelines + window，用 compute_pipeline_coverage 计算覆盖率。
    否则回退到旧逻辑 (coverage = 1.0 if events else 0.0)。

    Args:
        concurrent_events: {pid: [TraceEvent, ...]}
        pipelines: 完整 pipeline 数据 (可选)
        window_start: 窗口起始时间 ps (可选)
        window_end: 窗口结束时间 ps (可选)
        core_type: "veccore" 或 "cubecore"
    """
    state = ConcurrentState()

    scalar_pid = PIPELINE_NAME_MAP.get("SCALAR", PID_SCALAR)
    scalarldst_pid = PIPELINE_NAME_MAP.get("SCALARLDST", PID_SCALARLDST)
    mte2_pid = PIPELINE_NAME_MAP.get("MTE2", PID_MTE2)
    mte3_pid = PIPELINE_NAME_MAP.get("MTE3", PID_MTE3)
    mte1_pid = PIPELINE_NAME_MAP.get("MTE1", PID_MTE1)
    cube_pid = PIPELINE_NAME_MAP.get("CUBE", PID_CUBE)
    fixpipe_pid = PIPELINE_NAME_MAP.get("FIXPIPE", PID_FIXPIPE)

    use_coverage = (pipelines is not None and window_end > window_start)

    if use_coverage:
        # 共享 pipeline
        state.scalar_coverage = compute_pipeline_coverage(
            pipelines, scalar_pid, window_start, window_end)
        state.scalarldst_coverage = compute_pipeline_coverage(
            pipelines, scalarldst_pid, window_start, window_end)
        state.mte2_coverage = compute_pipeline_coverage(
            pipelines, mte2_pid, window_start, window_end)
        state.cachemiss_active = has_cachemiss_in_window(
            pipelines, window_start, window_end)
        state.flowctrl_active = has_flowctrl_in_window(
            pipelines, window_start, window_end)
        # veccore 专有
        state.mte3_coverage = compute_pipeline_coverage(
            pipelines, mte3_pid, window_start, window_end)
        # cubecore 专有
        if core_type == "cubecore":
            state.mte1_coverage = compute_pipeline_coverage(
                pipelines, mte1_pid, window_start, window_end)
            state.cube_coverage = compute_pipeline_coverage(
                pipelines, cube_pid, window_start, window_end)
            state.fixpipe_coverage = compute_pipeline_coverage(
                pipelines, fixpipe_pid, window_start, window_end)
    else:
        # 旧逻辑: 有事件 → coverage=1.0
        if scalar_pid in concurrent_events:
            state.scalar_coverage = 1.0
        if scalarldst_pid in concurrent_events:
            state.scalarldst_coverage = 1.0
        if mte2_pid in concurrent_events:
            state.mte2_coverage = 1.0
        if mte3_pid in concurrent_events:
            state.mte3_coverage = 1.0
        if mte1_pid in concurrent_events:
            state.mte1_coverage = 1.0
        if cube_pid in concurrent_events:
            state.cube_coverage = 1.0
        if fixpipe_pid in concurrent_events:
            state.fixpipe_coverage = 1.0

    # 保留 ops 列表
    if scalar_pid in concurrent_events:
        state.scalar_ops = concurrent_events[scalar_pid]
    if scalarldst_pid in concurrent_events:
        state.scalarldst_ops = concurrent_events[scalarldst_pid]
    if mte2_pid in concurrent_events:
        state.mte2_ops = concurrent_events[mte2_pid]
    if mte3_pid in concurrent_events:
        state.mte3_ops = concurrent_events[mte3_pid]
    if mte1_pid in concurrent_events:
        state.mte1_ops = concurrent_events[mte1_pid]
    if cube_pid in concurrent_events:
        state.cube_ops = concurrent_events[cube_pid]
    if fixpipe_pid in concurrent_events:
        state.fixpipe_ops = concurrent_events[fixpipe_pid]

    return state
