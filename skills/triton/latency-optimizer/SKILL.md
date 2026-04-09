---
name: latency-optimizer
description: >
  擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
  按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点，
  确保优化前后功能一致、精度一致。
  ⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。
argument-hint: >
  输入：code-file-path（代码文件路径）。
  输出：优化后的 Triton 代码、功能一致性说明、精度一致性说明。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Latency Optimizer Skill

<role>
你是一个擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
你的任务是按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点。
**必须确保优化前后的功能一致性和精度一致性。**
**⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。**
</role>

## 优化点执行顺序

Agent 必须严格按照以下顺序逐一检查优化点，**每次只能尝试一个优化点，命中后参考对应文档**。

⚠️ **前置要求**：必须先命中某个优化点的「命中条件」（代码特征满足典型代码特征之一且适用条件成立），才能加载对应的参考文档。未命中则跳过，禁止加载参考文档。

### 优化点 1：入参静态化优化

**适用条件**：代码中存在可声明为 `tl.constexpr` 的固定参数

**典型代码特征**：
```python
@triton.jit
def kernel(A, B, C, M, N,
            stride_am, stride_an,  # 运行时不变化的固定值
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr):
```

**判断逻辑**：
- 如果代码中存在运行时不变化的固定参数（如 stride、固定数值、BLOCK_SIZE等）未声明为 `tl.constexpr` → 涉及
- 如果所有固定参数都已正确声明为 `tl.constexpr` → 不涉及，跳过

**命中条件**：代码特征满足上述典型代码特征之一，且适用条件成立

**参考文档**：`references/constexpr_parameters.md`

---

### 优化点 2：Tiling 优化

**适用条件**：处理多维张量（3D 及以上）的规约类（Reduction）或归一化类（Normalization）算子，且规约轴（Reduction Axis）并非内存布局中的最连续轴（通常为最后一维 $N$）。

**典型代码特征**：
```python
@triton.jit
def kernel(input_ptr, output_ptr, dim1, dim2, ...):
    # 特征 1：向量化偏移 tl.arange 作用在非连续轴（如 dim1/M 轴）
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    # 特征 2：访存偏移计算中，向量化部分乘上了较大的 stride
    input_offset = m_offsets * stride_m + n_idx * stride_n
    # 特征 3：循环内部频繁进行还原操作（如 tl.sum）将向量压缩为标量
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    ...
    total_sum = tl.sum(acc, axis=0)
```

**判断逻辑**：
- 检查 `tl.load` 的偏移量计算：如果 `tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 涉及
- 检查循环累加器：如果累加器在还原轴上分块，但访存模式导致了非连续内存读取 → 涉及
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是最后一张量的最后一维），且实现了合并访存 → 不涉及，跳过

**命中条件**：代码逻辑旨在对某维度进行还原，但其分块策略导致硬件执行了跨步访存（Strided Access），未能利用硬件向量单元的合并访存特性。

**参考文档**：`references/tiling_optimization.md`

---

## 优化流程

```
1. 按顺序检查优化点
2. 对于当前优化点，先判断是否命中（代码特征满足 + 适用条件成立）：
   - 未命中 → 跳过，检查下一优化点
   - 命中 → 参考对应文档，应用优化策略
3. 应用优化后，必须加载 references/checklist.md 检查代码规范
4. 如果代码规范不满足 → 修改代码直到满足规范
5. 代码规范满足后 → 返回优化后的代码
```

**重要约束**：
- ⚠️ **只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式**
- ⚠️ **必须先命中优化点的「命中条件」，才能加载参考文档；未命中则跳过**
- 一次优化迭代只能使用一个优化点
- 一次只能参考一个文档

## 优化验证规则

**⚠️ 强制要求：在进行任何精度验证或性能验证之前，必须先执行 checklist 检查，确保所有代码规范都已满足。验证流程如下：**

1. **Checklist 检查**：加载 `references/checklist.md`，逐项检查代码是否满足所有规范要求
2. **不满足规范** → 修改代码直到满足所有规范要求，然后重新执行 checklist 检查确认
3. **满足规范后** → 执行精度验证和性能验证

- **成功**：优化后的性能不劣化（speedup ≥ 1.0），该优化结果作为下一次优化迭代的基线
- **失败**：优化后的性能劣化（speedup < 1.0），放弃本次优化结果，以优化前的代码作为下一次优化迭代的基线

## 参考资料索引

| 文档类型 | 文档路径 |
|----------|----------|
| 入参静态化优化 | `references/constexpr_parameters.md` |
| Tiling 优化 | `references/tiling_optimization.md` |
| 代码规范检查 | `references/checklist.md` |
