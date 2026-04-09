# 术语表

SIMT v2.0 项目中使用的主要术语和定义。

---

## A

**Active Mask**
: SIMT 执行中当前活跃线程的位掩码

---

## B

**BasicBlock**
: 控制流图的基本单位，单一入口和单一出口的连续指令序列

**Branch Target**
: 分支指令跳转的目标地址（标签）

---

## C

**CFG (Control Flow Graph)**
: 控制流图，表示程序执行路径的有向图

**Convergence Point**
: 收敛点，所有 divergent 路径重新汇合的点

---

## D

**Divergence**
: SIMT warp 内线程执行不同路径的情况

**Dominators**
: 支配点 - 从 entry 到某节点的所有路径必须经过的节点

---

## E

**Entry Block**
: CFG 的入口基本块（PC=0）

**Exit Block**
: CFG 的出口基本块

---

## I

**Immediate Post-Dominator**
: 立即后支配点 - 最近的 post-dominator

---

## P

**Post-Dominator**
: 后支配点 - 从某节点到 exit 的所有路径必须经过的节点

**PTX**
: Parallel Thread Execution, NVIDIA GPU 汇编语言

---

## R

**Reconvergence PC**
: 分支收敛点 PC，SIMT Stack 用于判断何时 reconverge

---

## S

**SIMT (Single Instruction Multiple Threads)**
: 单指令多线程，GPU 执行模型

**SIMT Stack**
: SIMT 栈，用于管理 divergent branch 的收敛

---

## W

**Warp**
: NVIDIA GPU 的基本调度单位（通常 32 个线程）

---

**维护**: 持续更新  
**最后更新**: 2026-04-11  
**版本**: 1.0
