---
name: "b_matrix_format_optimization"
description: "Optimizes B matrix memory layout for GEMM operations with dot instructions. Invoke when code contains tl.dot() and needs matrix multiplication performance optimization."
---

# B Matrix Format Optimization

## 1. Skill Overview

### 1.1 What This Skill Does
This skill optimizes the memory layout of the B matrix in GEMM (General Matrix Multiply) operations by reorganizing data into a block-based format. This optimization improves memory access patterns and cache utilization, particularly effective on NPU hardware.

### 1.2 When to Invoke This Skill
Invoke this skill when:
- The code contains `tl.dot()` instructions (this is a **prerequisite**)
- You need to optimize matrix multiplication performance
- The B matrix dimensions are compatible with the tile sizes used in the kernel

## 2. Prerequisites

**Step 1: Check for dot instruction**

Before applying this optimization, verify that the code contains a dot instruction:

```python
# Example of dot instruction in Triton
acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
```

If no dot instruction is present, this optimization is not applicable.

## 3. Implementation Steps

### Step 2: Identify Tile Variables

Identify the N-axis and K-axis tile variables in the code. These are typically defined as constants in the kernel function signature:

```python
@triton.jit
def gemm_kernel(
    # ...
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,  # This is the N-axis tile variable
    BLOCK_SIZE_K: tl.constexpr,  # This is the K-axis tile variable
    # ...
):
    # ...
```

Common naming conventions:
- N-axis: `BLOCK_SIZE_N`, `BLOCK_N`, `TILE_N`
- K-axis: `BLOCK_SIZE_K`, `BLOCK_K`, `TILE_K`

### Step 3: Convert B Matrix Format

Use the identified tile variables to convert the B matrix:

```python
def convert_b_matrix(b: torch.Tensor, block_k: int, block_n: int):
    """
    Convert B matrix to optimized block format.
    
    Args:
        b: Original B matrix with shape (K, N)
        block_k: K-axis tile size (e.g., BLOCK_SIZE_K)
        block_n: N-axis tile size (e.g., BLOCK_SIZE_N)
    
    Returns:
        Optimized B matrix with shape (k_div_block_k, n_div_block_n, block_k, block_n)
    """
    K, N = b.shape
    assert K % block_k == 0, f'K ({K}) must be divisible by block_k ({block_k})'
    assert N % block_n == 0, f'N ({N}) must be divisible by block_n ({block_n})'
    
    k_div_block_k = K // block_k
    n_div_block_n = N // block_n
    
    # Reshape and permute for optimized memory layout
    b_reshaped = b.view(k_div_block_k, block_k, n_div_block_n, block_n)
    b_optimized = b_reshaped.permute(0, 2, 1, 3).contiguous()
    
    return b_optimized
```

### Step 4: Update Kernel Access Pattern

Modify the kernel to access the optimized B matrix:

**⚠️ CRITICAL: Load A MUST be executed before Load B in the kernel loop.**

```python
# Inside the kernel
k_div_block_k = k // block_k
n_div_block_n = (block_n * BLOCK_SIZE_N) // block_n

# Calculate offset for the current block
b_offset = k_div_block_k * stride_b_k_div_block_k + n_div_block_n * stride_b_n_div_block_n

# IMPORTANT: Load A first (before loading B)
a_val = tl.load(a_ptr, boundary_check=(0, 1))

# Then load B
block_elements = block_k * block_n
b_raw = tl.load(b + b_offset + tl.arange(0, block_elements))

# Reshape to the required dimensions
b_val = b_raw.reshape(block_k, block_n)
```

## 4. Complete Example

### 4.1 Kernel Definition

```python
@triton.jit
def gemm_kernel(
    a, b, c, M, N, K,
    stride_am, stride_ak,
    stride_b_k_div_block_k, stride_b_n_div_block_n,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # ... (block calculation logic)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # ⚠️ CRITICAL: Load A MUST be before Load B
        a_val = tl.load(a_ptr, boundary_check=(0, 1))
        
        # Optimized B matrix access (after loading A)
        k_div_block_k = k // BLOCK_SIZE_K
        n_div_block_n = (block_n * BLOCK_SIZE_N) // BLOCK_SIZE_N
        
        b_offset = k_div_block_k * stride_b_k_div_block_k + n_div_block_n * stride_b_n_div_block_n
        b_raw = tl.load(b + b_offset + tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_N))
        b_val = b_raw.reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        
        # Dot instruction (prerequisite)
        acc += tl.dot(a_val, b_val, out_dtype=tl.int32)
        
        a_ptr = tl.advance(a_ptr, (0, BLOCK_SIZE_K))
```

### 4.2 Host Code with Autotune Config

**⚠️ CRITICAL: When calling `convert_b_matrix`, you MUST add a comment immediately before the function call to specify the actual tile variable names from the autotune config.**

```python
AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}),
]

def gemm(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K_from_b, N = b.shape
    
    # ========== CRITICAL: Must use this exact config structure ==========
    # Get the selected config from autotune
    # In practice, this would be determined by the autotuner
    config = AUTOTUNE_CONFIGS[0]  # Example: using first config
    BLOCK_SIZE_N = config.kwargs['BLOCK_SIZE_N']  # TileN = 256
    BLOCK_SIZE_K = config.kwargs['BLOCK_SIZE_K']  # TileK = 256
    # ================================================================
    
    # ========== CRITICAL: Add comment before convert_b_matrix ==========
    b_optimized = convert_b_matrix(b, block_k=BLOCK_SIZE_K, block_n=BLOCK_SIZE_N)
    # ====================================================================
    
    # Calculate strides for the optimized format
    k_div_block_k = K // BLOCK_SIZE_K
    n_div_block_n = N // BLOCK_SIZE_N
    
    stride_b_n_div_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N
    stride_b_k_div_block_k = n_div_block_n * stride_b_n_div_block_n
    
    # Launch kernel
    grid = (CORE_NUM,)
    gemm_kernel[grid](
        a, b_optimized, c, M, N, K,
        a.stride(0), a.stride(1),
        stride_b_k_div_block_k, stride_b_n_div_block_n,
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=config.kwargs['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
```

### 4.3 Example with Multiple Autotune Configs

When using multiple autotune configurations, the comment becomes even more important:

```python
AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 256}),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_kernel(...):
    # ... kernel implementation

def gemm(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K_from_b, N = b.shape
    
    # ========== CRITICAL: Must use this exact config structure ==========
    # After autotune selects a config, extract the tile sizes
    # Note: In practice, you would get the selected config from the autotuner
    # For demonstration, we use a specific config
    config = AUTOTUNE_CONFIGS[0]
    BLOCK_SIZE_N = config.kwargs['BLOCK_SIZE_N']  # TileN = 256
    BLOCK_SIZE_K = config.kwargs['BLOCK_SIZE_K']  # TileK = 256
    # ================================================================
    
    # ========== CRITICAL: Comment format ==========
    b_optimized = convert_b_matrix(b, block_k=BLOCK_SIZE_K, block_n=BLOCK_SIZE_N)
    # ==============================================
    
    # ... rest of the implementation
```

## 5. Key Benefits

### 5.1 Memory Access Efficiency
- **Contiguous memory access**: Loads entire blocks in a single operation
- **Improved cache utilization**: Block-based layout matches access patterns
- **Reduced memory transactions**: Fewer load operations for the same data

### 5.2 Hardware Optimization
- **NPU-friendly**: Matches hardware memory access patterns
- **Reduced addressing overhead**: Simplified offset calculations
- **Better parallelism**: Efficient data loading supports higher computational throughput

### 5.3 Flexibility
- **Dynamic tile sizes**: Adapts to any tile size defined in the kernel
- **No hardcoded constants**: Works with various block configurations
- **AutoTune compatible**: Can be combined with Triton's autotuning

## 6. Important Considerations

### 6.1 **⚠️ CRITICAL: Load Order Requirement**
**Load A MUST be executed before Load B in the kernel loop.**

This ordering is essential for:
- **Memory access optimization**: Ensures proper memory access patterns
- **Hardware efficiency**: Matches NPU hardware expectations
- **Correct execution**: Prevents potential race conditions or undefined behavior

**Correct order:**
```python
# ✓ CORRECT: Load A first
a_val = tl.load(a_ptr, boundary_check=(0, 1))
# Then load B
b_raw = tl.load(b + b_offset + tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_N))
```

**Incorrect order:**
```python
# ✗ INCORRECT: Load B before A
b_raw = tl.load(b + b_offset + tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_N))
a_val = tl.load(a_ptr, boundary_check=(0, 1))  # WRONG!
```

### 6.2 Dimension Requirements
- K must be divisible by the K-axis tile size (block_k)
- N must be divisible by the N-axis tile size (block_n)
- If dimensions don't meet requirements, consider padding

### 6.3 Memory Overhead
- The converted B matrix requires the same amount of memory
- Additional memory is needed during the conversion process
- Consider caching converted matrices for repeated use

### 6.4 Performance Trade-offs
- **One-time conversion cost**: Initial conversion takes time
- **Best for repeated operations**: Most beneficial when B matrix is reused
- **Batch processing**: Ideal for scenarios with multiple GEMM operations

### 6.5 **⚠️ CRITICAL: Comment Requirement**
**You MUST add a comment immediately before calling `convert_b_matrix` to specify:**
1. The actual variable name for `block_k` from the autotune config (e.g., `BLOCK_SIZE_K`)
2. The actual variable name for `block_n` from the autotune config (e.g., `BLOCK_SIZE_N`)

**Format:**
```python
# ========== CRITICAL: Comment format ==========
b_optimized = convert_b_matrix(b, block_k=<VARIABLE_NAME>, block_n=<VARIABLE_NAME>)
```

**Why this is important:**
- Ensures traceability between the conversion and the kernel configuration
- Makes it clear which autotune config is being used
- Helps with debugging and maintenance
- Prevents mismatched tile sizes between conversion and kernel execution

## 7. Troubleshooting

### 7.1 Common Issues

**Issue**: "K must be divisible by block_k"
- **Cause**: Matrix dimension doesn't match tile size
- **Solution**: Pad the matrix or adjust tile size

**Issue**: Incorrect results after optimization
- **Cause**: Incorrect stride calculations or mismatched tile sizes
- **Solution**: Verify stride formulas and ensure tile sizes match between conversion and kernel

**Issue**: No performance improvement
- **Cause**: Matrix too small or conversion overhead dominates
- **Solution**: Profile to identify bottleneck; consider matrix size

**Issue**: Missing comment before convert_b_matrix call
- **Cause**: Forgot to add the required comment
- **Solution**: Add comment specifying the autotune config variable names

**Issue**: Load B before Load A in kernel
- **Cause**: Incorrect load order in the kernel loop
- **Solution**: Ensure `tl.load(a_ptr)` is executed BEFORE `tl.load(b + b_offset + ...)`

### 7.2 Validation

Always validate the optimization produces correct results:

```python
# Test with small example
c_optimized = gemm(a, b_optimized, block_k=BLOCK_K, block_n=BLOCK_N)
c_reference = torch.matmul(a.float(), b.float()).half()

assert torch.allclose(c_optimized, c_reference, rtol=1e-3), "Results mismatch!"
```

## 8. Summary

This skill optimizes B matrix memory layout for GEMM operations by:
1. **Checking prerequisite**: Ensures code contains dot instruction
2. **Identifying tile variables**: Finds BLOCK_K and BLOCK_N from kernel definition
3. **Converting format**: Reorganizes B matrix to block-based layout
4. **Adding required comment**: Specifies autotune config variable names before conversion
5. **Updating access pattern**: Modifies kernel to use optimized loading
6. **Ensuring correct load order**: Load A MUST be executed before Load B in the kernel loop

The optimization is most effective when:
- Code contains `tl.dot()` instructions
- Matrix dimensions align with tile sizes
- B matrix is reused across multiple operations
- Running on hardware with specific memory access patterns (e.g., NPU)
- **Load A is executed before Load B in the kernel**

## 9. References

- [Triton Documentation](https://triton-lang.org/docs/)
- [Matrix Multiplication Optimization Guide](https://triton-lang.org/programming-guide.html)
- [NPU Programming Best Practices](https://developer.huawei.com/ict/en/site-type/docs)
