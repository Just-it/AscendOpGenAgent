# Proven Solutions Registry

## Purpose

Record successful optimization techniques discovered during evolution runs.
Enables cross-operator knowledge transfer: if a technique worked for operator A,
it may work for a similar operator B.

## Format

Each entry should follow this template:

```markdown
### {technique_name}

**Source**: {op_name} evolution, round {r}, parallel {p}
**Speedup**: {before}x → {after}x ({improvement_pct}% improvement)
**Operator type**: {elementwise | reduction | attention | matmul | ...}
**Hardware**: {chip_model}

**Technique description**:
{1-3 sentences describing what was done}

**Key code change**:
```cpp
// Before:
{key code snippet before optimization}

// After:
{key code snippet after optimization}
```

**Applicability conditions**:
- {condition 1}
- {condition 2}

**Strategy IDs**: {P1, P2, ...} or {X-series if novel}
```

## 按算子类型索引

| 算子类型 | 已有方案 | 来源 |
|---------|---------|------|
| attention | X1: Causal S2 Block Skipping | FlashAttentionSimple evolution |

_新方案沉淀时按此格式追加行_

## Recorded Solutions

### X1: Causal S2 Block Skipping (FlashAttentionSimple)

**Source**: FlashAttentionSimple evolution, arch_round8
**Speedup**: 1.003x → 1.83x (45.5% latency reduction at S=32K)
**Operator type**: attention
**Hardware**: Ascend 910B3

**Technique description**:
In causal attention, entire S2 blocks where all entries are masked (below the diagonal)
are skipped entirely — no cube matmul, no vector post-processing, no DMA.
The valid S2 range is computed per S1 block on the host side using the causal constraint:
`valid_s2_end = s1_end` (the last row of the current Q block determines the furthest valid K column).

**Key code change**:
```cpp
// Before: iterate over all S2 blocks
for (int s2 = 0; s2 < totalS2Blocks; s2++) { ProcessBlock(s1, s2); }

// After: skip invalid blocks
int validS2End = (s1BlockEnd) / blockSize + 1;
for (int s2 = 0; s2 < min(validS2End, totalS2Blocks); s2++) { ProcessBlock(s1, s2); }
```

**Applicability conditions**:
- Attention operator with causal (lower-triangular) mask
- Block-tiled computation (not element-wise attention)
- Sequence length > 512 (smaller sequences have few blocks to skip)

**Strategy IDs**: X-series (novel, discovered during evolution)

---

_Add new entries below as they are discovered._
