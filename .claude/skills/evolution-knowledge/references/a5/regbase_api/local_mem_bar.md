# LocalMemBar - VF 内流水同步

## 定义

```cpp
template <MemType src, MemType dst>
__simd_callee__ inline void LocalMemBar()
```

LocalMemBar 用于在 VF 函数内部同步不同流水线的内存访问，确保数据一致性。

## MemType 枚举

| 值 | 含义 |
|----|------|
| `VEC_STORE` | Vector 写 UB 流水 |
| `VEC_LOAD` | Vector 读 UB 流水 |
| `SCALAR_STORE` | Scalar 写 UB 流水 |
| `SCALAR_LOAD` | Scalar 读 UB 流水 |
| `VEC_ALL` | 所有 Vector 读写流水 |
| `SCALAR_ALL` | 所有 Scalar 读写流水 |

## 有效组合

| src → dst | 语义 |
|-----------|------|
| VEC_STORE → VEC_LOAD | 确保写完再读（RAW） |
| VEC_LOAD → VEC_STORE | 确保读完再写（WAR） |
| VEC_STORE → VEC_STORE | 确保写序（WAW） |
| VEC_STORE → SCALAR_LOAD | Vector 写完 Scalar 才能读 |
| VEC_STORE → SCALAR_STORE | Vector 写完 Scalar 才能写 |
| VEC_LOAD → SCALAR_STORE | Vector 读完 Scalar 才能写 |
| SCALAR_STORE → VEC_LOAD | Scalar 写完 Vector 才能读 |
| SCALAR_STORE → VEC_STORE | Scalar 写完 Vector 才能写 |
| SCALAR_LOAD → VEC_STORE | Scalar 读完 Vector 才能写 |
| VEC_ALL → VEC_ALL | 全同步（Vector 内部） |
| VEC_ALL → SCALAR_ALL | 全同步（Vector → Scalar） |
| SCALAR_ALL → VEC_ALL | 全同步（Scalar → Vector） |

## 使用场景

### 替代 A3 的 pipe_barrier
```cpp
// A3: pipe_barrier(PIPE_V)
// A5 等价:
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
```

### Store 后读取
```cpp
StoreAlign(ubAddr, dstReg, mask);
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();  // 确保写完
LoadAlign(srcReg, ubAddr);  // 安全读取
```

## 注意事项

- 编译器在 VF 融合时会自动优化同步指令（删除冗余、精确插入必要同步）
- 用户应优先依赖编译器自动同步，仅在明确需要时手动插入
- 手动过多的 LocalMemBar 会阻碍编译器 OOO（乱序执行）优化
