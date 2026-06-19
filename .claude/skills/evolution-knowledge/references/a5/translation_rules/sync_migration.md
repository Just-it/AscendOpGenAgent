# 同步机制迁移

## A3 → A5 同步机制对照

| A3 机制 | A5 替代 | 适用场景 |
|---------|--------|---------|
| `pipe_barrier(PIPE_V)` | `LocalMemBar<VEC_STORE, VEC_LOAD>()` | VF 内部 Vector 流水同步 |
| `pipe_barrier(PIPE_ALL)` | `LocalMemBar<VEC_ALL, VEC_ALL>()` | VF 内部全同步 |
| `SetFlag/WaitFlag` | 仍可用 + Mutex 增强 | VF 外部（__aicore__ 函数中） |
| 无 | `Mutex` | 核内异步流水精细同步 |
| `CrossCoreSetFlag/WaitFlag` | 增强版（支持 AIV0/AIV1 单独触发） | 核间同步 |

## VF 内部同步: LocalMemBar

### 替代 pipe_barrier

```cpp
// A3
pipe_barrier(PIPE_V);

// A5 (VF 内部)
AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
```

### 常用组合

```cpp
// Store 后读取（RAW 依赖）
StoreAlign(ubAddr, dstReg, mask);
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
LoadAlign(srcReg, ubAddr);

// Vector 写完 Scalar 读
StoreAlign(ubAddr, dstReg, mask);
LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_LOAD>();
// Scalar 操作...
```

### 注意事项
- 编译器在 VF 融合时会**自动优化**同步指令
- 过多手动 LocalMemBar 会阻碍 OOO 优化
- 优先依赖编译器自动同步

## VF 外部同步: SetFlag/WaitFlag

在 `__aicore__` 函数中，SetFlag/WaitFlag 仍然可用：

```cpp
// A3 和 A5 通用
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
```

## 新增: Mutex

Mutex 提供核内异步流水指令间的精细同步，类似 CPU 锁机制。

```cpp
// 锁定指定流水
Mutex::Lock(pipeType);
// 执行需要同步的操作
// ...
Mutex::Unlock(pipeType);
```

**适用场景**: 需要比 SetFlag/WaitFlag 更细粒度控制的场景。

## 新增: CrossCore 增强

### A3
```cpp
// AIV 统一触发 AIC 等待
CrossCoreSetFlag();
CrossCoreWaitFlag();
```

### A5
```cpp
// AIV0 和 AIV1 可单独触发 AIC 等待
// 支持 1:1 模式（单个 AIV 触发）
// 支持 1:2 模式（两个 AIV 分别触发）
CrossCoreSetFlag(mode);
CrossCoreWaitFlag(mode);
```

## 迁移策略

### Phase 1 (保守翻译)
1. VF 内部: 将 `pipe_barrier` 替换为对应的 `LocalMemBar`
2. VF 外部: 保持 `SetFlag/WaitFlag` 不变
3. 不引入 Mutex（Phase 2 优化时再考虑）

### Phase 2 (优化)
1. 减少不必要的 LocalMemBar（依赖编译器自动同步）
2. 评估是否用 Mutex 替代粗粒度 SetFlag/WaitFlag
3. 利用 CrossCore 增强模式优化核间通信
