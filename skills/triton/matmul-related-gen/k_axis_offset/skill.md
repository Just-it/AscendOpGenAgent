# K-Axis Offset Optimization Algorithm for GEMM

## Inputs

- `input_triton_code` - Original Triton kernel code containing matrix multiplication
  - Must contain `tl.dot()` operation (indicates matrix multiplication)
  - Should use `tl.make_block_ptr()` for pointer management
  - Should have K-axis reduction loop

## Outputs

- `triton_code_with_koffset` - Optimized Triton code with K-axis offset technique applied
  - All K positions are computed exactly once
  - Uses cyclic K-axis traversal

## Overview

K-Axis Offset Optimization is a technique for optimizing GEMM (General Matrix Multiply) operations on NPU by distributing K-axis computation across different blocks with different starting positions. This reduces memory access conflicts and improves parallelism.

## Preconditions

Before applying this optimization, verify the following conditions:

1. **Contains Matrix Multiplication**: The code must contain `tl.dot()` operation
2. **K-Axis Reduction Loop**: There should be a loop that iterates over the K dimension
3. **Block Pointer Usage**: Code should use `tl.make_block_ptr()` for pointer management
4. **Sufficient K Dimension**: K should be significantly larger than BLOCK_SIZE_K for meaningful benefit
5. **Multi-Core Target**: Target hardware should have multiple cores for parallelism benefit

## Code Pattern Recognition

### Pattern to Match

Look for the following code patterns that indicate applicability of K-axis offset optimization:

```python
# Standard K-axis loop pattern (BEFORE optimization)
for k in range(0, K, BLOCK_SIZE_K):
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))
```

### Key Indicators

1. **`tl.dot()` presence**: The code contains `tl.dot()` operation
2. **K-axis loop**: A loop iterating over K dimension with step size `BLOCK_SIZE_K`
3. **Pointer advancement**: Uses `tl.advance()` to move pointers along K dimension
4. **Accumulator pattern**: Uses an accumulator variable (e.g., `acc`) that accumulates dot products

### Detection Logic

```python
def can_apply_koffset(code):
    indicators = [
        "tl.dot(" in code,
        "tl.advance(" in code,
        "tl.make_block_ptr(" in code,
        any(keyword in code for keyword in ["BLOCK_SIZE_K", "for k", "range(0, K"])
    ]
    return sum(indicators) >= 3
```

## Algorithm Principle

### Core Idea

Different blocks start from different K positions and traverse the entire K axis cyclically:

1. **Calculate K Offset**: Each block calculates its starting K position using modulo arithmetic
2. **Cyclic Traversal**: Traverse from offset to K end, then wrap around to 0 and continue to offset
3. **Complete Coverage**: Ensure all K positions are computed exactly once

### Mathematical Foundation

For a block with index `block_idx`:
```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = (2 * block_idx) % num_k_blocks

for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
```

The K-axis traversal follows this pattern:
- **Phase 1**: From `k_offset * BLOCK_SIZE_K` to `K` (exclusive)
- **Phase 2**: From `0` to `k_offset * BLOCK_SIZE_K` (exclusive)

This ensures complete K-axis coverage with cyclic behavior.

## Implementation

### Method 1: Fusion Approach with Per-Iteration Pointer Creation (Recommended)

This approach is based on the reference implementation and provides an optimized fusion pattern:

```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = (2 * block_idx) % num_k_blocks

acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
    
    # Create new pointers with current offset each iteration
    a_ptr = tl.make_block_ptr(
        base=a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m * BLOCK_SIZE_M, current_k),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        base=b,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(current_k, block_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
```

**Pros**:
- Simplest code structure
- No complex pointer management
- Direct fusion pattern
- No conditional logic

**Cons**:
- Does not use `tl.advance()` (violates checklist requirement)
- More pointer creation overhead
- Potentially less efficient than hybrid approach

### Method 2: Two-Phase Approach (Alternative)

```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = ((2 * block_idx) % num_k_blocks) * BLOCK_SIZE_K

# Phase 1: From offset to K
a_ptr = tl.make_block_ptr(..., offsets=(..., k_offset), ...)
b_ptr = tl.make_block_ptr(..., offsets=(k_offset, ...), ...)

num_k_blocks_phase1 = tl.cdiv(K - k_offset, BLOCK_SIZE_K)
for k_iter in range(num_k_blocks_phase1):
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))

# Phase 2: From 0 to offset
a_ptr = tl.make_block_ptr(..., offsets=(..., 0), ...)
b_ptr = tl.make_block_ptr(..., offsets=(0, ...), ...)

num_k_blocks_phase2 = tl.cdiv(k_offset, BLOCK_SIZE_K)
for k_iter in range(num_k_blocks_phase2):
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))
```

**Pros**:
- Clear separation of two phases
- Uses `tl.advance()` efficiently (satisfies checklist requirement)

**Cons**:
- Two separate loops
- More code duplication
- Manual calculation of phase lengths

## Key Techniques

### 1. Modulo Arithmetic for Wrap-Around

```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = (2 * block_idx) % num_k_blocks

for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
```

This formula automatically calculates the current K position and handles wrap-around when exceeding K.

### 2. Hybrid Pointer Management

- **Primary**: Use `tl.advance()` for forward movement (efficient)
- **Secondary**: Use `tl.make_block_ptr()` for wrap-around (necessary)

### 3. Conditional Pointer Recreation

```python
if k_iter > 0 and current_k == 0:
    # Recreate pointers at position 0
```

Only recreate pointers when actually wrapping around to avoid overhead.

## Workflow

Follow these steps to apply K-axis offset optimization to your Triton code:

### Step 1: Analyze Input Code

- Check if the code contains `tl.dot()` operation
- Identify the K-axis loop structure
- Locate `tl.make_block_ptr()` and `tl.advance()` usage
- Extract matrix dimensions (M, N, K) and BLOCK_SIZE_K

### Step 2: Extract Parameters

- Extract K dimension from the code
- Identify the block indexing scheme (e.g., `block_idx`, `pid`, `pgid`)
- Find the current K-loop boundaries

### Step 3: Apply Optimization

1. **Calculate K Offset**: Add `num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)` and `k_offset = (2 * block_idx) % num_k_blocks`
2. **Initialize Accumulator**: Create zero-initialized accumulator
3. **Modify Loop Structure**: Change from simple K-loop to cyclic traversal
4. **Implement Chosen Method**:
   - **Method 1 (Fusion Approach)**: Create new pointers per iteration with current_k
   - **Method 2 (Two-Phase)**: Use two separate loops with tl.advance()
5. **Handle Wrap-Around**: Ensure complete K-axis coverage

### Method Selection Guide

- **Method 1 (Fusion Approach)**: Recommended - optimized for fusion with simplest code structure
- **Method 2 (Two-Phase)**: Alternative - clearer separation when needed

## Use Cases

### When to Use K-Axis Offset

1. **Large K Dimension**: When K is significantly larger than BLOCK_SIZE_K
2. **Memory-Bound Operations**: When memory access is the bottleneck
3. **Multi-Core Systems**: When utilizing multiple NPU cores
4. **Regular GEMM**: Standard matrix multiplication without special patterns

### When NOT to Use

1. **Small K Dimension**: When K is close to BLOCK_SIZE_K
2. **Irregular Access Patterns**: When K-axis access is already optimized
3. **Single-Core Execution**: No benefit with single core

## Implementation Checklist

- [ ] Calculate `k_offset` using `k_offset = (2 * block_idx) % num_k_blocks`
- [ ] Initialize pointers appropriately based on chosen method
- [ ] Implement cyclic K-axis traversal
- [ ] Handle wrap-around correctly
- [ ] Ensure complete K-axis coverage
- [ ] Verify correctness with allclose test

### Method-Specific Requirements

#### Method 1 (Fusion Approach)
- [ ] Create new pointers per iteration
- [ ] Use current_k directly for offsets

#### Method 2 (Two-Phase)
- [ ] Calculate phase lengths manually
- [ ] Use `tl.advance()` for both phases

## Common Pitfalls

### 1. Missing Wrap-Around Logic

**Wrong**:
```python
for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
    # Always advance, even when wrapping around
    a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
```

**Correct**:
```python
for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
    
    if k_iter > 0 and current_k == 0:
        # Recreate pointers at 0
        a_ptr = tl.make_block_ptr(..., offsets=(..., 0), ...)
    
    # Only advance when not wrapping
    if k_iter < num_k_blocks - 1:
        a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
```

### 2. Incorrect Offset Calculation

**Wrong**:
```python
k_offset = block_idx * BLOCK_SIZE_K  # Can exceed K
```

**Correct**:
```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = (2 * block_idx) % num_k_blocks  # Always within [0, num_k_blocks)
```

### 3. Not Using tl.advance()

**Wrong** (violates checklist):
```python
for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
    a_ptr = tl.make_block_ptr(..., offsets=(..., current_k), ...)
    # Always recreate pointers
```

**Correct** (satisfies checklist):
```python
# Use advance for forward movement
a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
# Only recreate when necessary
if k_iter > 0 and current_k == 0:
    a_ptr = tl.make_block_ptr(..., offsets=(..., 0), ...)
```

## Integration with Existing Code

### Step-by-Step Integration

1. **Identify K-loop**: Find the K-axis loop in your GEMM kernel
2. **Calculate Offset**: Add `num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)` and `k_offset = (2 * block_idx) % num_k_blocks`
3. **Initialize Accumulator**: Create zero-initialized accumulator
4. **Modify Loop Structure**: Change from simple K-loop to cyclic traversal
5. **Implement Fusion Approach**: Create new pointers per iteration with current_k
6. **Test**: Verify correctness and performance

### Example Integration

**Before** (standard GEMM):
```python
for k in range(0, K, BLOCK_SIZE_K):
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
    a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
    b_ptr = tl.advance(b_ptr, (BLOCK_SIZE_K, 0))
```

**After** (K-axis offset - Fusion Approach):
```python
num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
k_offset = (2 * block_idx) % num_k_blocks
acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

for k_iter in range(num_k_blocks):
    k_idx = (k_iter + k_offset) % num_k_blocks
    current_k = k_idx * BLOCK_SIZE_K
    
    # Create new pointers with current offset each iteration
    a_ptr = tl.make_block_ptr(
        base=a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m * BLOCK_SIZE_M, current_k),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        base=b,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(current_k, block_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    a_val = tl.load(a_ptr, boundary_check=(0, 1))
    b_val = tl.load(b_ptr, boundary_check=(0, 1))
    acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
```

## Advanced Topics

### Tuning Parameters

1. **BLOCK_SIZE_K**: Larger values reduce loop iterations but may increase register pressure
2. **CORE_NUM**: Should match hardware core count for optimal parallelism
3. **Offset Granularity**: Can adjust offset calculation for different access patterns

### Performance Results

Experimental results on M=128, N=4096, K=7168 matrix multiplication:

| Configuration | K offset Calculation | Task Duration (us) | Performance Improvement |
|--------------|---------------------|-------------------|------------------------|
| Without K offset | Standard K-loop | 69.98 | Baseline |
| With K offset | `(2 * block_idx) % num_k_blocks` | 62.32 | **11% faster** |

The modulo operation with multiplier 2 in the offset calculation provides better performance by reducing memory access conflicts.

### Combining with Other Optimizations

K-axis offset can be combined with:
- Split-K parallelization
- Double buffering
- Pipeline optimization
- Memory coalescing

### Debugging Tips

1. **Print current_k**: Verify wrap-around behavior
2. **Check coverage**: Ensure all K positions are visited
3. **Validate offsets**: Confirm pointer positions are correct
4. **Monitor performance**: Compare with baseline implementation

## References

- Triton Programming Guide: https://triton-lang.org/main/index.html
- Ascend NPU Documentation: https://ascend.github.io/triton-ascend/
- GEMM Optimization Techniques: See `reference/operator_examples/` directory

## Version History

- **v1.0** (2026-04-14): Initial version with two-phase approach
- **v1.1** (2026-04-14): Unified loop with hybrid pointer management
- **v1.2** (2026-04-14): Added comprehensive documentation and examples
- **v1.3** (2026-04-14): Updated to use fusion approach as recommended method
- **v1.4** (2026-04-25): Updated to use `(2 * block_idx) % num_k_blocks` formula for offset calculation

## License

This optimization technique is provided for use in Triton-Ascend projects. Please refer to the project license for usage terms.