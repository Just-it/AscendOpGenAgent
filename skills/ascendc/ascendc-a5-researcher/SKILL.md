---
name: ascendc-a5-researcher
description: >
  Research optimization opportunities for A5 AscendC kernels through expert code
  diff analysis and bounded structural exploration guided by profiling data.
argument-hint: >
  Required: kernel source path, output_dir, bottleneck report.
  Optional: expert reference code, baseline profile.
---

# A5 AscendC Researcher

Analyze A5 AscendC kernel performance and find optimization opportunities through:
1. **Expert code diff** — compare expert reference code against current kernel
2. **Bounded structural search** — enumerate alternatives along 5 dimensions
3. **Profiling-guided diagnosis** — use msprof + grounding chains to identify bottlenecks

## When to Use

- After pattern matching (Stage 2 of migration) is exhausted
- When expert provides reference code for comparison
- When target GPU/NPU ratio is not met and no known pattern applies

## Input

- Kernel source file (e.g., `{output_dir}/kernel/{op}_kernel.h`)
- `{output_dir}/bottleneck_diagnosis.md`
- `{output_dir}/performance_report.md`
- (Optional) Expert reference code for diff analysis
- (Optional) Baseline profile for delta comparison

## File Access Boundary

You may read ONLY:
- The provided kernel source file
- `{output_dir}/**`
- `../a5-shared-references/**`

You MUST NOT read:
- `archive_tasks/**`
- `../ascendc-translator/**`
- `../tilelang-designer/**`
- `../performance-analyzer/**`
- `../trace-recorder/**`
- Any non-A5 skill directories

Do NOT treat A3 / TileLang skill content or historical tasks as "expert reference implementations" for A5 research.

## Required References

ALWAYS load:
```text
../a5-shared-references/ASCENDC_LANGUAGE_REFERENCE.md
../a5-shared-references/ROOFLINE_MODEL.md
../a5-shared-references/MSPROF_AGENT_GUIDE.md
../a5-shared-references/exploration/GROUNDING_CHAINS.md
../a5-shared-references/exploration/STRUCTURAL_DIMENSIONS.md
../a5-shared-references/exploration/EXPLORATION_PROTOCOL.md
../a5-shared-references/PLATFORM_BUGS.md
```

## Protocol

### Step 1: Profile the Bottleneck

Run msprof locally on the worst-performing case:
```bash
rm -rf /tmp/msprof_out && \
  MSPROF=/usr/local/Ascend/cann-9.0.T501/tools/profiler/bin/msprof && \
  export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.T501/x86_64-linux/lib64:$LD_LIBRARY_PATH && \
  $MSPROF --output=/tmp/msprof_out -- python3 utils/performance.py {output_dir} ascendc
```

Extract key metrics: `aiv_vec_ratio`, `aiv_mte2_ratio`, `aiv_scalar_ratio`, `task_duration`.

### Step 2: Match Grounding Chains

Load `../a5-shared-references/exploration/GROUNDING_CHAINS.md` and match observed metrics:

| Chain | Trigger | Points To |
|-------|---------|-----------|
| GC-1 | All pipes < 30% | D1 (loop order), D2 (work granularity) |
| GC-2 | vec > 90%, scalar < 5% | D1, D5 (tiling) |
| GC-3 | scalar > 20% | D2 (persistent), D3 (cache preload) |
| GC-4 | mte2 > 50% | D3 (prefetch depth), D1 (data reuse) |
| GC-5 | No pipe saturated | D4 (sync), D3 (queue depth) |
| GC-6 | Large gap small vs large | D2 (persistent kernel) |
| GC-7 | N items share GM read | D1 (sort-to-reuse) |

### Step 3: Enumerate Alternatives

For each matched dimension, enumerate the 3-4 concrete alternatives from `../a5-shared-references/exploration/STRUCTURAL_DIMENSIONS.md`.

Focus areas:
- loop order / data reuse improvements
- persistent / runtime dispatch suitability
- TQue depth / preload / cache strategy
- synchronization granularity
- tile parameter matching

### Step 4: Filter Already-Tried

Cross-reference with:
- Current kernel code (do not suggest patterns already in use)
- `../a5-shared-references/patterns/PATTERN_INDEX.md` (remove already-applied patterns)
- Previous optimization directives in `{output_dir}/` (avoid repeating failed directions)

### Step 5: Formulate Hypotheses

For each remaining alternative, write a structured hypothesis:

```text
HYPOTHESIS: H{N}
  Dimension: D1/D2/D3/D4/D5
  Change: {concrete code change}
  Grounding: {which msprof metric supports this}
  Prediction: {which case category will improve and by how much}
  Falsification: {what result would prove this wrong}
  Cost: {low/medium/high}
  UB Budget: {must be < 192KB total}
```

Each hypothesis MUST pass a UB budget check before any implementation recommendation.

### Step 6: Rank and Report

Rank hypotheses by (predicted improvement / cost). Report top 3 to the caller.

For each ranked hypothesis, provide:
- Priority order
- Expected impact on the worst-performing case
- Estimated implementation effort
- Precision risk assessment

## Bounding Rules

Read `../a5-shared-references/exploration/EXPLORATION_PROTOCOL.md` for full rules:
- **Max 3 structural changes** per campaign (D5 sweeps don't count)
- **Max 90 min** wall-clock
- **Early termination**: 2 consecutive regressions → STOP
- **Hopeless case detector**: if all pipes > 80% AND ratio matches core count ratio → escalate to human
- **Never sacrifice precision for performance**
- If profiling shows the kernel is already near roofline (>60% efficiency), state clearly that "further optimization gains are limited"

## Expert Code Diff

When expert provides reference code:
1. Load both files, identify structural differences
2. Classify each difference by dimension (D1-D5)
3. For each difference, check if it maps to a known pattern
4. For novel differences, formulate as exploration hypothesis
5. Prioritize expert-suggested changes over blind search

## Output

Return a structured report with:
1. msprof analysis summary
2. Grounding chain matches
3. Ranked hypothesis list (max 3 structural + unlimited D5)
4. Recommended execution order
5. Total estimated budget

Write findings to `{output_dir}/research_report.md` in a machine-readable format so the generator or main agent can consume it directly.

## Anti-Hack Rules

- NEVER fabricate bottlenecks without profiling evidence
- NEVER suggest "rewrite with a completely different algorithm" — stay within bounded structural changes
- NEVER suggest wrapping CANN built-in operators as an "optimization"
- NEVER relax precision requirements to improve benchmark numbers
- NEVER read msprof binary trace files (GB-scale, will crash context)
