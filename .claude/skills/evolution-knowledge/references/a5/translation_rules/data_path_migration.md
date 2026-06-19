# 数据通路迁移

## 删除的通路

### 1. L1 Buffer → GM

**影响**: DataCopy 不再支持 L1 → GM 直接搬运。

**替代方案**:
- 纯 Cube 场景: Mmad → L0C → Fixpipe → GM
- 需要经过 UB 的场景: L1 → UB → GM

```cpp
// A3: 直接搬运
DataCopy(gmTensor, l1Tensor, count);

// A5: 通过 Fixpipe
FixpipeParamsV220 fixpipeParams;
fixpipeParams.nSize = nSize;
fixpipeParams.mSize = mSize;
fixpipeParams.srcStride = srcStride;
fixpipeParams.dstStride = dstStride;
Fixpipe<float, float, CFG_ROW_MAJOR>(gmTensor, l0cTensor, fixpipeParams);
```

### 2. GM → L0A/L0B

**影响**: LoadData 不再支持 GM → L0A/L0B 直接搬运。

**替代方案**: 拆分为两步
```cpp
// A3: 一步
LoadData(l0aTensor, gmTensor, loadParams);

// A5: 两步
// Step 1: GM → L1
DataCopy(l1Tensor, gmTensor, nd2nzParams);
// Step 2: L1 → L0A
LoadData(l0aTensor, l1Tensor, loadDataParams);
```

## 新增的通路

### 1. UB ↔ L1 Buffer (双向)
```cpp
// UB → L1
DataCopy(l1Tensor, ubTensor, count);
// L1 → UB
DataCopy(ubTensor, l1Tensor, count);
```

### 2. L0C → UB (通过 Fixpipe)
```cpp
// Fixpipe 输出到 UB（而非 GM）
Fixpipe<float, float, CFG_ROW_MAJOR>(ubTensor, l0cTensor, fixpipeParams);
```

### 3. ND-DMA
```cpp
// 多维数据搬运
SetLoopModePara(loopParams);
DataCopy(dstTensor, srcTensor, ndDmaParams);
```

### 4. Fixpipe NZ2DN
```cpp
// L0C → GM 带格式转换 (NZ → DN)
Fixpipe<float, float, CFG_NZ2DN>(gmTensor, l0cTensor, fixpipeParams);
```

## L0A 分形变化: ZZ → NZ

### A3 (ZZ 分形)
```cpp
// L1 (NZ) → L0A 需要 NZ→ZZ 分形转换
for (int i = 0; i < mBlocks; ++i) {
    LoadData2DParams params;
    params.repeatTimes = kBlocks;
    params.srcStride = mBlocks;
    params.ifTranspose = false;
    LoadData(l0aLocal[dstOffset], l1Local[srcOffset], params);
    srcOffset += 16 * 16;
    dstOffset += k * 16;
}
```

### A5 (NZ 分形)
```cpp
// L1 (NZ) → L0A 不需要分形转换
LoadData2DParams params;
params.repeatTimes = m * k / 512;  // 小 z 矩阵个数
params.srcStride = 1;
params.dstGap = 0;
params.ifTranspose = false;
LoadData(l0aLocal, l1Local, params);
```

## SetLoadDataBoundary 删除

### A3
```cpp
SetLoadDataBoundary(boundaryValue);  // 自动绕回
```

### A5: 手动拆分 LoadData 进行绕回
```cpp
// 手动多次调用 LoadData
LoadData(a2, leftMatrix, loadData3dParams);
LocalTensor<T> a3 = a2[256];
LoadData(a3, leftMatrix, loadData3dParams);
LocalTensor<T> a4 = a2[512];
LoadData(a4, leftMatrix, loadData3dParams);
```

## 迁移检查清单

- [ ] 检查所有 DataCopy 调用，确认不使用 L1→GM 通路
- [ ] GM→L0A/L0B 的 LoadData 拆分为两步
- [ ] 更新 Fixpipe 参数适配新通路
- [ ] 移除 SetLoadDataBoundary，改为手动绕回
- [ ] L0A 分形参数从 ZZ 调整为 NZ
- [ ] 验证 ND-DMA 场景是否可简化搬运逻辑
