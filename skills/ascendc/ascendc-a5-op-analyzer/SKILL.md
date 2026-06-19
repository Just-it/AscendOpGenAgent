---
name: ascendc-a5-op-analyzer
description: >
  Analyze source operator and plan AscendC implementation: accepts CUDA (.cu/.cuh),
  PyTorch (.py + .json from NPUKernelBench), or AscendC from other chips.
  Algorithm classification, pattern matching, mandatory audits, SIMT/SIMD decision,
  and test specification generation.
argument-hint: >
  Required: source (file path — .cu/.cuh for CUDA, .py for PyTorch, .h for AscendC port).
  Optional: kernel_name, target_shapes, source_gpu (default a100), json_spec (companion .json for PyTorch).
---

# Source → AscendC Migration Analyzer

You are an AscendC migration expert. Analyze the source operator and produce a
complete implementation plan. **Do NOT generate AscendC code** — only analysis and strategy.

## File Access Boundary

You may read ONLY:
- `source`
- `{output_dir}/**`
- `../a5-shared-references/**`
- The benchmark source file when applicable

You MUST NOT read:
- `archive_tasks/**`
- `../ascendc-translator/**`
- `../tilelang-designer/**`
- `../performance-analyzer/**`
- `../trace-recorder/**`
- Any non-A5 skill directories

## Step 0: Input Detection

Detect source type from file extension and content:

```
1. Read source file from $ARGUMENTS
2. Detect type:
   - .cu / .cuh → SOURCE_TYPE = CUDA
   - .py with "class Model(nn.Module)" → SOURCE_TYPE = PYTORCH
     - Look for companion .json: same directory, same numeric prefix (e.g., 13_Cat.json for 13_Cat.py)
   - .h / .cpp with "#include <kernel_operator.h>" but no "__simt_vf__" → SOURCE_TYPE = ASCENDC_PORT
   - Otherwise → ask user to clarify
3. Print: "Detected source type: {SOURCE_TYPE}"
```

For CUDA input, proceed to "CUDA Analysis Path" (Step 2).
For PyTorch input, proceed to "PyTorch Analysis Path" (Step 2P).
For AscendC port, proceed to "AscendC Port Path" (Step 2A) [NOT YET IMPLEMENTED — ask user].

## Inputs

- `source`: file path from $ARGUMENTS (CUDA .cu/.cuh, PyTorch .py, or AscendC .h)
- `kernel_name`: primary kernel function name (optional for PyTorch — inferred from Model.forward)
- `target_shapes`: runtime shape configs to evaluate (optional — read from .json for PyTorch)
- `source_gpu`: source GPU platform (default: a100, only used for CUDA path)
- `json_spec`: companion .json test case file (auto-detected for PyTorch)

## Step 1: Load Knowledge (selective)

```
ALWAYS load:
  ../a5-shared-references/patterns/PATTERN_INDEX.md        (~80 lines, pattern routing)
  ../a5-shared-references/hardware/INDEX.md                (~50 lines, key hw deltas)
  ../a5-shared-references/hardware/target/ascend950pr.md   (~120 lines, NPU specs)
  ../a5-shared-references/SIMT_VS_SIMD_DECISION.md         (P-P9 decision framework — MANDATORY for Step 5)

THEN load source GPU doc:
  ../a5-shared-references/hardware/source/{source_gpu}.md  (~80 lines)
  If file not found: web search "{source_gpu} specs" → create stub

DO NOT load domain pattern files yet (Step 3 will select them).
```

## Step 2P: PyTorch Analysis Path (when SOURCE_TYPE = PYTORCH)

Skip Steps 2/3/4 and use this path instead.

### 2P.1 Parse PyTorch Model

Read the `.py` file. Extract:
- `Model(nn.Module).forward()` method signature and body
- What PyTorch ops are called (e.g., `torch.cat`, `F.layer_norm`, `torch.cumsum`)
- Input types: tensor vs attribute (dim, eps, normalized_shape, etc.)
- Output: single tensor, tuple, or list

### 2P.2 Parse JSON Test Cases

Read companion `.json` file (JSONL format, one test case per line). Extract:
- All input tensor shapes and dtypes (fp32, fp16, bf16)
- Attribute values and their ranges
- Variable-length inputs (e.g., `tensor_list` with 2-5 tensors for Cat)
- **Broadcasting detection**: For multi-input ops, compare input shapes. If any pair
  has different shapes, flag as broadcast case. Count same-shape vs broadcast cases.
  Broadcasting MUST be handled inside the kernel (stride-aware reads), NOT by
  `expand_as().contiguous()` in the Python wrapper — that materializes the expansion
  in HBM and is unfair vs CANN which handles broadcast natively via strided access.
- Generate `test_spec.md` directly from JSON:
  - Group by dtype, list all shapes
  - Separate same-shape vs broadcast cases with counts
  - Count: N dtypes × M shape groups = K total test cases
  - Precision thresholds: use NPUKernelBench standard (atol=1e-2, rtol=1e-2)

### 2P.3 Algorithm Classification (from PyTorch ops)

Classify into ONE primary class based on the PyTorch operation:

| PyTorch Op | Class | Typical Decision |
|------------|-------|-----------------|
| `torch.add/mul/sub/abs` | **elementwise** | SIMD |
| `F.gelu/silu/relu/sigmoid` | **elementwise** | SIMD |
| `torch.cat` | **data_movement** | SIMT (variable inputs, stride calc) |
| `torch.permute + contiguous` | **data_movement** | SIMT (stride remapping) |
| `torch.split/chunk` | **data_movement** | SIMT |
| `F.pad` | **data_movement** | SIMT (boundary logic) |
| `tensor.repeat` | **data_movement** | SIMT |
| `F.layer_norm/group_norm` | **reduction** | SIMD (vectorized reduce) or SIMT |
| `torch.sum/mean` | **reduction** | SIMD or SIMT depending on dim |
| `torch.cumsum` | **scan** | SIMT |
| `torch.sort` | **sort** | SIMT |
| `torch.topk` | **selection** | SIMT |
| `torch.index_select/gather` | **gather** | SIMT |
| `torch.scatter_add` | **scatter_add** | SIMT |
| `torch.histc` | **histogram** | SIMT |
| `torch.nonzero` | **compaction** | SIMT |

If the op doesn't match any pattern, describe its data flow and classify by dominant access pattern.

### 2P.4 Proceed to Step 5 (SIMT/SIMD Decision)

After 2P.3 classification, go directly to Step 5 for the SIMT/SIMD decision.
Then Step 6 (test spec — already generated in 2P.2), then Step 7 (outputs).

## Step 2: Analyze CUDA Kernel (when SOURCE_TYPE = CUDA)

Read the CUDA source. For the primary kernel, extract:

### 2.1 Signature
- Function name, template params, all parameters with types
- `__global__` / `__device__` decorators
- Launch config if visible (grid, block dimensions)

### 2.2 Algorithm Classification
Classify into ONE primary class:
| Class | Indicator | Example |
|-------|-----------|---------|
| **scatter_add** | atomicAdd in loop over indirect indices (write to arr[index[i]]) | Pooling backward |
| **gather** | indirect read (arr[index[i]]), direct write | SG forward, Pooling forward |
| **elementwise** | direct read + direct write, no indirect indexing | init, clear |
| **reduction** | warp shuffle, block reduce, shared memory accumulate | grad_weight |
| **hash_table** | atomicCAS lock/unlock, bucket scan, linear probing | HKV ops |
| **cooperative** | __shfl, __ballot, cooperative_groups | tile-based ops |

### 2.3 Thread Model
- How threadIdx.x maps to work items
- Multi-dimensional decomposition (e.g., index × embedding, token × hidden)
- Fixed constants (BRE, TILE_SIZE, WARP_SIZE)

### 2.4 Data Access Pattern
- Sequential vs random reads
- Scatter vs direct writes
- Per-element vs accumulated output

## Step 3: Load Relevant Patterns

Based on classification from Step 2, load ONLY relevant domain files:
```
if class in [scatter_add, reduction]:
    load ../a5-shared-references/patterns/domains/scatter_add.md
if has multi-dim decomposition:
    load ../a5-shared-references/patterns/domains/thread_utilization.md
if has memory access optimization opportunity:
    load ../a5-shared-references/patterns/domains/memory_access.md
ALWAYS load:
    ../a5-shared-references/patterns/domains/precision.md
    ../a5-shared-references/patterns/domains/kernel_launch.md
```

For each loaded pattern, mark: **MANDATORY** (severity HIGH+) or **RECOMMENDED**.

## Step 4: Mandatory Audits (NEVER SKIP — CUDA path only)

**For PyTorch path**: Skip Steps 3 and 4 entirely — PyTorch has no CUDA-specific
patterns to audit. The test_spec is already generated from JSON in Step 2P.2.
Proceed directly to Step 5.

### 4.0 PyTorch vs CUDA Consistency Check (OL-28, CRITICAL)
**PyTorch is the spec. CUDA is just one implementation that may have bugs.**

If PyTorch source is available:
1. Run PyTorch reference on test data (CPU or GPU)
2. Run CUDA kernel on same data
3. Compare bit-by-bit
4. If PyTorch ≠ CUDA → document all discrepancies in `pytorch_cuda_diff.md`
5. AscendC must align with **PyTorch**, not CUDA

If PyTorch source is NOT available:
1. Flag as "CUDA-only migration — PyTorch consistency unverified"
2. Treat CUDA as best-available reference (not ground truth)

**Edge case discovery checklist** (OL-29):
- Extreme value combinations in same group/block (e.g., 1e38 + 1.0 in MX format)
- Integer overflow in bit-shift operations (`1 << n` where n > 30)
- Rounding boundaries (values at exact quantization level boundaries)
- Zero, -0, subnormal, inf, nan handling
- Saturation (max representable + 1 ULP)

Output: `pytorch_cuda_diff.md` (mandatory when PyTorch source exists)

### 4.1 int64 Truncation Audit
Scan ALL parameters and local variables:
- Any int64_t / long long / int64 parameter → MUST preserve in AscendC
- Flag every `(int)int64_var` or `static_cast<int>(int64_var)`
- Check multiplications: `index * dim` where either operand could exceed INT32_MAX
- **Rule: judge by interface type, NEVER by current test data range**
Output: `int64_audit.md`

### 4.2 Scatter-Add Detection
- Find atomicAdd/atomicCAS inside loops with indirect write targets
- If detected: flag for sorted-edge variant (P-P21) as RECOMMENDED
- Estimate fan-in if possible (edges per output node)
Output: `scatter_add_analysis.md`

### 4.3 Thread Utilization Analysis (TUA, static, no msprof needed)
For each dimension in the thread decomposition:
```
utilization = min(actual_work[dim], assigned_threads[dim]) / assigned_threads[dim]
```
If utilization < 50% for ANY target shape → flag for runtime dispatch (P-P20)
Output: `thread_utilization.md`

## Step 5: SIMT vs SIMD Decision

**Load**: `../a5-shared-references/SIMT_VS_SIMD_DECISION.md` (complete decision framework with case studies)

Apply decision tree (summary — see reference for full version):
```
Step 1: atomicAdd / scatter-write? → SIMT
Step 2: indirect indexing (arr[index[i]])? → SIMT
Step 3: group-local dependency (group_size < tile_size)?
  group_size < 256? → SIMT (per-group loop kills SIMD parallelism)
  group_size >= 256? → SIMD candidate
Step 4: computation fully vectorizable (same op per element)?
  YES → SIMD (MTE2/VEC pipeline overlap)
  NO  → SIMT (per-element heterogeneous ops)
```

⚠️ CRITICAL (OL-30): SIMD optimization MUST NOT sacrifice precision.
    Skipping per-group processing for SIMD performance = wrong precision.
    Validated: MXFP4 SIMD "fast" beats SIMT but breaks spec.

## Step 6: Generate Test Specification

Write `test_spec.md` with:
- dtype × shape matrix: fp32, fp16, bf16 × all target shapes
- Precision thresholds: fp32 atol=1e-4, fp16 atol=1e-2, bf16 atol=2e-2
- Edge cases: dim=1, dim=3, edges=0, edges=1 (adapt to kernel semantics)
- Expected waivers: scatter-add ops may have fp16/bf16 mismatch (not a bug)
- GPU golden ref generation commands

## Step 7: Output Files

Write to `{output_dir}/`:

**CUDA path outputs:**
```
analysis.md              - Algorithm analysis + thread decomposition
migration_strategy.md    - SIMT/SIMD decision + rationale + target patterns
int64_audit.md          - int64 truncation findings
scatter_add_analysis.md  - Scatter-add detection results
thread_utilization.md    - TUA per shape
pattern_match.md         - Applicable patterns (MANDATORY/RECOMMENDED/N/A)
test_spec.md            - Full test matrix
```

**PyTorch path outputs:**
```
analysis.md              - Algorithm analysis (from PyTorch ops classification)
migration_strategy.md    - SIMT/SIMD decision + rationale + target patterns
pattern_match.md         - Applicable patterns (MANDATORY/RECOMMENDED/N/A)
test_spec.md            - Full test matrix (generated from .json)
```

## Step 8: Human Checkpoint

Present to user:
1. Algorithm class + SIMT/SIMD decision (1 paragraph)
2. Mandatory patterns to apply (table)
3. Audit findings (int64, scatter-add, TUA)
4. Test matrix size (N dtype × M shapes = K tests)

Ask: "Approve this migration plan? [Y/modify/abort]"

## Anti-Hack Rules

- NEVER skip int64 audit (lesson: flagged 5 times, "confirmed safe" 3 times without code change)
- NEVER skip TUA even if "shapes look fine" (lesson: dim=9 had 28% utilization, not obvious)
- NEVER load ALL patterns — only load domains matching the classification
- NEVER generate AscendC code in this skill — that's the Generator's job
