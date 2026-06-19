# Hardware Loop vs Software Loop

## 概述

A5 编译器能将符合条件的循环编译为 **Hardware Loop**（硬件循环），比 Software Loop（软件循环）更高效。

## Hardware Loop 编码要求

### 1. 迭代变量类型
所有 VF 中的循环迭代变量必须为 **`uint16_t`** 类型。

### 2. 起始值和步长
- 起始值**必须为 0**
- 步长**必须为 1**

### 3. 循环体内无跳转指令
- **不允许** `if/else`
- **不允许** 三元运算符 `?:`
- 编译器会尝试消除 if/else 但不保证

### 4. 循环次数不可变
- 循环边界一旦确定不能在循环体内修改

## 示例

### [OK] Hardware Loop（符合条件）
```cpp
for (uint16_t i = 0; i < loopBound; i++) {           // 单层
    for (uint16_t j = 0; j < loopBound; j++) {       // 双层
        for (uint16_t k = 0; k < loopBound; k++) {   // 三层
            for (uint16_t m = 0; m < loopBound; m++) { // 四层（最多）
                // 计算操作
            }
        }
    }
}
```

### [BAD] Software Loop（不符合条件）
```cpp
// 原因: 包含 if 语句
for (uint16_t i = 0; i < loopBound; i++) {
    if (condition) { /* ... */ }
}

// 原因: 起始值不为 0，步长不为 1
for (uint16_t i = 2; i < loopBound * 3; i += 2) {
    // ...
}

// 原因: 使用了非 uint16_t 类型
for (uint32_t i = 0; i < loopBound; i++) {
    // ...
}
```

## if/else 替代方案

### 方案1: `if constexpr`（编译期常量）
```cpp
if constexpr (CONDITION) {
    // 编译期决定，无运行时开销
}
```

### 方案2: `for(1)` 替代 if
```cpp
// [BAD] 原始: if 语句阻止 Hardware Loop
uint16_t tailK = srcK % floatRepSize;
uint16_t hasTail = !!tailK;
if (tailK > 0) {
    // 尾块处理
}

// [OK] 优化: for(1) 替代
uint16_t tailK = srcK % floatRepSize;
uint16_t hasTail = !!tailK;
for (uint16_t i = 0; i < hasTail; i++) {
    // 尾块处理 — 编译器可优化为 Hardware Loop
}
```

## 循环边界注意事项

如果使用外层循环计数器作为内层循环边界，先转存到新变量：

```cpp
// [BAD] 直接使用外层循环变量（可能影响 Hardware Loop 识别）
for (uint16_t i = 0; i < outer; i++) {
    for (uint16_t j = 0; j < i; j++) { ... }  // 边界是外层变量
}

// [OK] 先转存
for (uint16_t i = 0; i < outer; i++) {
    uint16_t bound = i;
    for (uint16_t j = 0; j < bound; j++) { ... }
}
```
