# PTX 指令集架构参考手册（PTX ISA 9.1）

> 本手册基于 [NVIDIA 官方 PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)（PTX ISA 9.1），系统整理了所有核心指令类别，适用于 CUDA 开发者、编译器工程师及 GPU 底层性能优化人员。  
> **最后更新**：2026 年 1 月

---

## 目录结构

#### 整数指令
- [算术与数据移动指令总览](9.7.1_integer_arith.md)

#### 整数扩展精度
- [扩展精度整数指令](9.7.2_integer_extended.md)

#### 浮点与混合精度
- [浮点与混合精度指令](float_9.7.3-5.md)

#### 比较与选择
- [比较与选择指令](cmp_sel_9.7.6-7.md)

#### 位操作 
- [逻辑与移位指令](bitwise_9.7.8.md)

### 数据移动与转换
- [9.7.9 数据移动与转换指令](ptx_9.7.9_data_movement_conversion.md)

### 纹理与表面内存
- [9.7.10–9.7.11 纹理与表面指令](ptx_9.7.10-9.7.11_texture_surface.md)

### 控制流
- [9.7.12 控制流指令](ptx_9.7.12_control_flow.md)

### 同步与通信
- [9.7.13 同步与通信指令](ptx_9.7.13_synchronization_communication.md)

### 栈、视频与杂项
- [9.7.17–9.7.19 栈操作、视频指令与杂项](ptx_9.7.17-9.7.19_stack_video_misc.md)

---


## 使用说明

- 所有链接文档均以 **Markdown** 格式提供，可直接在 GitHub/GitLab 或本地 Markdown 阅读器中查看。
- 每篇子文档包含：
  - 指令语法与语义
  - 支持的数据类型与限定符
  - 架构要求（sm_xx）
  - 典型代码示例
  - 最佳实践与性能提示
- 推荐配合 [NVIDIA PTX ISA 9.1 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 使用。

---

## 文件清单（供仓库组织参考）

```text
ptx_isa_handbook/
├── OVERVIEW.md                          # 本文档（总览）
├── ptx_9.7.2_extended_precision_integers.md
├── ptx_9.7.3-9.7.5_floating_point.md
├── ptx_9.7.6-9.7.7_comparison_and_selection.md
├── ptx_9.7.8_logic_and_shift.md
├── ptx_9.7.9_data_movement_conversion.md
├── ptx_9.7.10-9.7.11_texture_surface.md
├── ptx_9.7.12_control_flow.md
├── ptx_9.7.13_synchronization_communication.md
└── ptx_9.7.17-9.7.19_stack_video_misc.md
```

> ✅ **提示**：点击上方目录中的链接即可跳转到对应章节。  
> 📚 **版权**：内容基于 NVIDIA PTX ISA 9.1，遵循 [NVIDIA 软件许可协议](https://docs.nvidia.com/cuda/eula/index.html)。