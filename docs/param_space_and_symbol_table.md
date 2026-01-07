# PTX-EMU 参数空间建立与符号表构建流程

## 概述

本文档详细描述了PTX-EMU中参数空间建立和符号表构建的机制，这是PTX模拟器中非常重要的一个环节，确保PTX指令可以正确访问到内核参数。

## 整体流程

在PTX模拟器中，参数空间建立和符号表构建是执行PTX内核的关键步骤，主要涉及以下几个环节：

1. 内核启动 (`launchPtxInterpreter`)
2. 函数解释器初始化 (`funcInterpreter`)
3. 参数空间设置 (`setupKernelArguments`)
4. 符号表构建和映射

## 详细流程分析

### 1. 启动内核

```cpp
void PtxInterpreter::launchPtxInterpreter(...) {
    // 初始化指令工厂，设置各种上下文
    this->kernelArgs = args;  // 从外部传入的参数指针数组
    this->param_space = nullptr; // 初始化param_space为空
    // ...
    funcInterpreter(name2Sym, label2pc); // 调用函数解释器
    // 内核执行结束后释放参数空间
}
```

### 2. 函数解释器初始化

```cpp
void PtxInterpreter::funcInterpreter(...) {
    setupConstantSymbols(name2Sym);    // 设置常量符号
    setupKernelArguments(name2Sym);    // 设置内核参数（重点）
    setupLabels(label2pc);             // 设置标签
    // ...
}
```

### 3. 参数空间建立流程 (`setupKernelArguments`)

这是核心部分，具体步骤如下：

1. **计算参数总大小**：
   ```cpp
   size_t total_param_size = 0;
   for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
       auto e = kernelContext->kernelParams[i];
       total_param_size += Q2bytes(e.paramType) * (e.paramNum ? e.paramNum : 1);
   }
   ```

2. **分配参数空间**：
   ```cpp
   if (total_param_size > 0) {
       this->param_space = MemoryManager::instance().malloc_param(total_param_size);
       memset(this->param_space, 0, total_param_size);
   }
   ```

3. **参数数据复制和符号表构建**：
   ```cpp
   size_t offset = 0;
   for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
       auto e = kernelContext->kernelParams[i];
       Symtable *s = new Symtable();
       // 设置符号信息
       s->name = e.paramName;
       s->elementNum = e.paramNum;
       s->symType = e.paramType;
       s->byteNum = Q2bytes(e.paramType);
       
       size_t param_size = s->byteNum * (e.paramNum ? e.paramNum : 1);
       
       // 将参数值拷贝到PARAM空间
       if (this->param_space != nullptr) {
           memcpy((char *)this->param_space + offset, kernelArgs[i], param_size);
       }
       
       // 在符号表中存储参数空间中该参数的地址
       s->val = (uint64_t)((char *)this->param_space + offset);
       
       name2Sym[s->name] = s;  // 将符号添加到符号表
       offset += param_size;
   }
   ```

## 关键机制分析

### 参数空间管理

- **内存分配**：使用`MemoryManager::instance().malloc_param()`分配连续的参数空间
- **数据复制**：将从CUDA运行时传入的`kernelArgs[i]`复制到分配的参数空间中
- **偏移计算**：按顺序将各个参数放置在参数空间的不同偏移位置

### 符号表映射机制

- **符号创建**：为每个参数创建一个[Symtable](file:///mnt/ubuntu/chisuhua/github/PTX-EMU/include/ptxsim/common_types.h#L10-L17)对象
- **地址映射**：`s->val`存储参数在参数空间中的实际地址
- **名称映射**：`name2Sym[s->name] = s`建立参数名到符号表对象的映射

## 数据流向分析

```
外部参数(kernelArgs) 
    ↓ (memcpy)
参数空间(param_space) 
    ↓ (地址记录在s->val)
符号表(name2Sym) 
    ↓ (PTX指令执行时查询)
PTX指令处理器
```

## 实际应用流程

当PTX指令需要访问参数时：

1. 指令解析器查找参数名在[name2Sym](file:///mnt/ubuntu/chisuhua/github/PTX-EMU/thread_context.h#L58-L58)符号表中
2. 获取对应的[Symtable](file:///mnt/ubuntu/chisuhua/github/PTX-EMU/include/ptxsim/common_types.h#L10-L17)对象
3. 从`Symtable::val`获取实际的内存地址
4. 从该地址读取或写入数据

这样就实现了PTX指令能够正确访问到传入的参数数据。

## 内存管理

- 参数空间在内核执行结束后通过`MemoryManager::instance().free_param()`释放
- 避免了内存泄漏问题

## 设计要点

这个设计巧妙地将CUDA运行时传递的参数转换为PTX模拟器内部可以访问的连续内存空间，并通过符号表建立了参数名称与内存地址的映射关系，使得PTX指令可以像访问本地变量一样访问内核参数。

根据项目规范，PARAM空间的内存管理需要遵循以下要点：

1. 符号表（name2Sym）中应存储参数的地址（即`kernelArgs[i]`的地址），具体为指向PARAM空间内地址（param_space + offset），而非参数值本身
2. 当执行`.param`内存空间的LD/ST指令时，需对存储的地址进行解引用，获取实际的参数值
3. 这种设计保持了与CUDA运行时参数传递机制的一致性，确保PTX指令如`ld.param.u64 %rd1, [symbol]`能正确获取参数值而非参数地址
4. PARAM空间内存应在内核执行结束后立即释放，推荐在launchPtxInterpreter函数末尾或PtxInterpreter析构函数中进行清理
5. 必须确保PARAM空间的分配与释放成对出现，防止内存泄漏
6. PARAM空间的管理应作为PtxInterpreter类的成员责任，使用类成员变量（如param_space）跟踪其状态