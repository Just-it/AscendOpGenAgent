# VF 融合规则

## 融合三阶段

### 1. 浅融合 (Shallow Fusion)
- 编译器分析两个 VF 的**控制流等价性**
- 构建 Cost Model 判断融合收益
- 将 VF 外部控制流合并到 VF 内部
- 将 VF 外部 Software Loop 固化为 VF 内部 Hardware Loop

### 2. 深融合 (Deep Fusion)
- 继续融合 VF 内部的 Hardware Loop
- 减少 Hardware Loop 启动开销
- **大幅减少冗余 Load/Store 操作**
- 最大化寄存器复用

### 3. VF 内自动同步
- 编译器精确插入必要同步指令
- 删除冗余同步指令
- 最大化 OOO（乱序执行）能力

## 融合条件

### 控制流等价
两个 VF 必须具有**相同的控制流结构**才能融合：
- 相同的循环次数和嵌套结构
- 相同的条件分支结构（最好没有条件分支）

### 融合前检查
编译器执行以下检查：
1. 两个 VF 是否控制流等价
2. Main-side 中间代码是否能在 VF 内执行
3. 融合后是否有正收益（无寄存器溢出、代码体积可接受）

## 融合友好编码模式

### [OK] 推荐: 连续计算 API 模式
```cpp
__aicore__ inline void Compute()
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::DataCopy(yLocal, xLocal, count);
    AscendC::Add(yLocal, xLocal, xLocal, count);   // 简单连续调用
    AscendC::Mul(yLocal, yLocal, xLocal, count);    // 编译器识别后自动融合
    outQueueY.EnQue<T>(yLocal);
}
```

### [BAD] 避免: 高维切分 API 模式
```cpp
__aicore__ inline void Compute()
{
    uint64_t mask = 128;
    AscendC::Add(yLocal, xLocal, xLocal, mask, 4, {1,1,1,8,8,8});   // 复杂参数
    AscendC::Mul(yLocal, yLocal, xLocal, mask, 4, {1,1,1,8,8,8});   // 编译器难以融合
}
```

### [OK] 推荐: 多个独立 VF 函数
```cpp
// DivVF 和 AddVF 控制流等价 → 编译器自动融合
DivVF(dstAddr, srcAddr, count, repeatTime, oneRepNum);
AddVF(dstAddr, dstAddr, count, repeatTime, oneRepNum);
```

### [OK] 推荐: 局部变量传参
```cpp
// 将成员变量/结构体字段提取为局部变量
uint16_t srcK = tiling.srcK;
uint16_t srcM = tiling.srcM;
for (uint16_t i = 0; i < srcM; i++) {
    ReduceMax(maxAddr + i * reduceK, srcAddr + i * srcK, workAddr, originK);
}
```

### [BAD] 避免: 直接访问成员变量
```cpp
// 直接访问 tiling.srcK 会导致融合失败
for (uint16_t i = 0; i < (uint16_t)tiling.srcM; i++) {
    ReduceMax(maxAddr + i * FLOAT_NUM_PER_BLK,
              srcAddr + i * tiling.srcK, workAddr, (uint16_t)originalSrcShape.k);
}
```

### [OK] 推荐: 循环不变量外提
```cpp
__simd_vf__ inline void DuplicateVF(__ubuf__ T* dstAddr, T scalarValue,
    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    RegTensor<T> dstReg;
    MaskReg mask = CreateMask<T>();
    Duplicate(dstReg, scalarValue);         // 循环外执行一次
    for (uint16_t i = 0; i < repeatTimes; i++) {
        StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);  // 循环内只做存储
    }
}
```

## 融合策略

编译器采用 **"尽可能融合"** 策略。用户应：
1. 使用连续计算 API（非高维切分模式）
2. 确保多个 VF 控制流等价
3. 避免在 VF 中使用条件分支
4. 使用局部变量而非成员变量
5. 将循环不变计算提到循环外
6. 控制寄存器数量不超过限制（32 RegTensor, 8 MaskReg）
