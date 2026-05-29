---
name: Matrix Multiplication (matmul) Diagonal Tiling Core Logic Algorithm
description: Achieve efficient memory access through diagonal core mapping rules
---

## Code Generation Requirements:
1. Core mapping rule: Ensure adjacent block_id access different M, N blocks; tl.swizzle2d is not allowed
2. Generated code must satisfy CORE_Load_Balancing
3. Generated code must satisfy Coding_Rules
4. The algorithm must include at least 4 autotune parameters: BLOCK_M, BLOCK_N, BLOCK_K, and GROUP_SIZE

## Autotune Config Generation Rules:
1. Keep all original BLOCK_M, BLOCK_N, BLOCK_K configurations from the base code
2. For each original configuration, generate variants with GROUP_SIZE=4 and GROUP_SIZE=8
3. Total configs = original_configs × len(GROUP_SIZE_values)
4. GROUP_SIZE values: [4, 8]

### Example:
Original configs (3):
- Config A: {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}
- Config B: {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256}
- Config C: {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256}

After applying diagonal tiling (6 configs):
- Config A + GROUP_SIZE=4: {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 4}
- Config A + GROUP_SIZE=8: {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}
- Config B + GROUP_SIZE=4: {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 4}
- Config B + GROUP_SIZE=8: {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 8}
- Config C + GROUP_SIZE=4: {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 4}
- Config C + GROUP_SIZE=8: {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}

## Related Variable Definitions:

### block_id Mapping Rule
- block_id=idx means block_id is assigned to a Core ID, for example: block_id=tl.program_id(0)
- (M,N) represents the 2D Tile block index in M and N directions
- block_id=0 → (0,1) means Core 0 processes the Tile block at position (0,1)

### GROUP_2D Definition
- GROUP_2D contains GROUP_SIZE rows, each row has GROUP_SIZE data blocks
- The total number of data blocks in a GROUP_2D is: tiles_per_group = GROUP_SIZE * GROUP_SIZE

## Algorithm Flow
1. Each group has GROUP_2D small data blocks; diagonal tiling within each group, taking GROUP_SIZE=4 as an example:
    - First 4*4 diagonal tiling group:
      block_id=0 → (0,0), block_id=1 → (1,1), block_id=2 → (2,2), block_id=3 → (3,3)
      block_id=4 → (0,1), block_id=5 → (1,2), block_id=6 → (2,3), block_id=7 → (3,0)
      block_id=8 → (0,2), block_id=9 → (1,3), block_id=10 → (2,0), block_id=11 → (3,1)
      block_id=12 → (0,3), block_id=13 → (1,0), block_id=14 → (2,1), block_id=15 → (3,2)
    - Second 4*4 diagonal tiling group:
      block_id=16 → (0,4), block_id=17 → (1,5), block_id=18 → (2,6), block_id=19 → (3,7)

## Reference Implementation
```python
# Get current program ID (Core ID)
pid = tl.program_id(0)

# Calculate grid dimensions
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)

# Calculate total virtual blocks, padded by Group
# To implement diagonal tiling, we divide the grid into GROUP_SIZE x GROUP_SIZE SuperBlocks
groups_m = tl.cdiv(num_pid_m, GROUP_SIZE)
groups_n = tl.cdiv(num_pid_n, GROUP_SIZE)
num_groups = groups_m * groups_n

# Number of blocks per Group
tiles_per_group = GROUP_SIZE * GROUP_SIZE

# Total virtual tasks (including padded blocks)
total_virtual_tiles = num_groups * tiles_per_group

# Loop through tasks assigned to current Core
for v_idx in range(pid, total_virtual_tiles, CORE_NUM):
    # 1. Decode v_idx into Group ID and in-Group ID
    group_idx = v_idx // tiles_per_group
    in_group_idx = v_idx % tiles_per_group
    
    # 2. Calculate Group coordinates (grid of groups)
    group_m = group_idx // groups_n
    group_n = group_idx % groups_n
    
    # 3. Calculate in-Group coordinates (Diagonal Tiling)
    # Mapping rule:
    # block_id=0 -> (0,0), block_id=1 -> (1,1) ...
    # i = local_id % GROUP_SIZE
    # j = local_id // GROUP_SIZE
    # m = i
    # n = (i + j) % GROUP_SIZE
    
    i = in_group_idx % GROUP_SIZE
    j = in_group_idx // GROUP_SIZE
    local_m = i
    local_n = (i + j) % GROUP_SIZE
    
    # 4. Calculate global coordinates
    block_m = group_m * GROUP_SIZE + local_m
    block_n = group_n * GROUP_SIZE + local_n
    
    # 5. Boundary check: use if instead of continue, as Triton compiler may not support continue
    if block_m < num_pid_m and block_n < num_pid_n:
        # --- Start matrix multiplication calculation ---
```
