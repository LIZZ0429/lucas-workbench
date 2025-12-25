你的这份 最终 NetMamba-FSCIL 执行计划质量非常高，说明你已经把两篇论文和对应实现都吃透了。我现在从 工程、理论一致性、模块边界、可训练性、可运行性 五个角度逐项检查，指出：

✔ 哪些内容完全正确

⚠ 哪些地方需要轻微调整

❌ 哪些地方必须修正否则会报错或训练不稳定

以下是逐条审查和优化建议。

✅ 总体评价：你的迁移计划已经接近“正式可实现版本”

尤其是以下几点非常正确：

完全删除了 SS2D（这是绝对正确的）

明确了 NetMamba 仅用作 Backbone encoder

g_base/g_inc 两个分支保留且变为 1D 版本

ETFHead 需要改维度 （这点经常被忽略，你写得对）

Memory 和 Loss 都从 Mamba-FSCIL 保留

分阶段训练逻辑准确

目录结构组织非常工程化

整体可以说：这是正确的 NetMamba-FSCIL 设计蓝图。

🔍 逐项审查（重点检查错误风险点）
✔ 2.1 模块迁移与修改计划——整体正确

但下面几点需要特别注意：

⚠（重要）需要修正 1：MambaNeck 输入不是“灰度图像”，而是 NetMamba 输出的特征维度

你写了：

修改MLP投影层以适配灰度图像输入（1通道）

这是错误的。

✔ 正确版本：

Dual Selective SSM Projector（g_base/g_inc）的输入不是图像，也不是“1通道”。
它接收的是 NetMamba 输出的特征序列形状：

[batch, seq_len, embed_dim]


例如：

[32, 400, 256]


所以你应该写的是：

✔ 正确写法（修正）：

修改 MLP 投影层输入维度，以适配 NetMamba 的输出 embedding 维度（如 256 或 512）。

不能按“灰度图 1 通道”去适配——那是图像领域用法。

⚠ 需要修正 2：ETF Head 的输入不是 256，而是 projector 输出的维度

你写：

修改输入特征维度以适配NetMamba输出

严格来说，你应该写：

✔ 正确：

ETFHead 输入 = Projector 输出
而不是 = NetMamba 输出

因为 projector 会映射一次：

NetMamba → g_base/g_inc → fused feature → L2 → ETFHead


如果 projector 输出的维度改变，ETFHead 输入也要跟着变。

⚠ 需要修正 3：StrideEmbed 和 Block 迁移路径有轻微问题

你写了：

迁移到 models/backbones/mamba_utils.py

建议：

models/backbones/netmamba/
    ├── modules.py     # StrideEmbed, Block, create_block
    ├── netmamba.py    # NetMamba backbone


理由：
这些是 NetMamba 内部模块，不应该和 Mamba-FSCIL 的 neck/utils 混在一起。

保持模块清晰性非常重要，避免多处重复 import 或命名冲突。

⚠ 需要修正 4：Dual Selective SSM Projector 需要使用 NetMamba 的 MambaBlock 而不是 Mamba-FSCIL 的版本

你写：

修改为1D Mamba

这是正确的，但需要进一步明确：

✔ 最佳做法：

直接使用 NetMamba 中的 create_block（1D Mamba）作为基础 SSM

结合 Mamba-FSCIL 的 gating / f_z / f_B / f_Δ 逻辑

而不是：

用 Mamba-FSCIL 中的 old SSM 实现（它性能不如 NetMamba）

也不能复用 SS2D（你已经删除，这是正确的）

你的计划里没有明确这一点。
为了避免跑出来效果差，必须明确：

✔ 正确写法：

Projector（g_base/g_inc）内部应当使用 NetMamba 的 create_block 构造的 1D MambaBlock，以保持 backbone 与 neck 的 Mamba 内核一致。

⚠ 需要修正 5：Memory Module 的抽取不能简单“抽出 classifier 中的代码”

Mamba-FSCIL 的 memory 更新逻辑是：

在 session 0 中，计算所有 base 类的 feature mean

存入 memory

在增量 session t：

对新类计算 feature mean

append 到 memory

使用 memory 做 NCM 分类和支持 ETFHead

你需要确保清楚：

✔ 正确版本：

Memory 不依赖任何图像结构，可以抽取，但必须保证其：

保存的维度 = Projector 输出维度

更新逻辑每 session 重计算类特征中心

评估时使用 memory 中的 class mean 构建 ETFHead prototype

你的计划看起来是正确的，只是需要特别强调：

Memory 与图像无关，但必须重新实现均值更新逻辑适配 NetMamba 的 embedding。

✔ 其余部分检查（全部正确）

以下部分都没有问题：

训练流程（base session vs incremental session）

任务配置系统

损失函数迁移并保持不变

评估指标（AVG / PD / FINAL）

目录结构规划

实现顺序规划

所有关于 SS2D 删除的内容

非常优秀。

🎯 最终修正后的关键修改点（最重要部分）

以下是必须在你的计划中修订的关键点，我帮你重新总结一版（你可以直接复制回你的文档）：

✅ 必须修正的四个点（精简版）
1. MambaNeck 输入不是灰度图 1 通道，而是 NetMamba 输出维度

正确：

projector input_dim = netmamba.output_dim

2. ETFHead 输入不是 NetMamba 输出，而是 projector 输出

正确：

etf_input_dim = projector_output_dim

3. Projector 的 1D Mamba 应采用 NetMamba 的 MambaBlock

理由：性能更高，与 backbone 内核一致。

4. Memory 中存的类中心维度 = projector 输出维度

必须保持一致。

🎉 总结（确认结果）

你的计划已经很成熟，但我帮你修正了 4 个核心错误点。
这些错误如果不改，会导致：

维度不匹配

无法训练或性能严重下降

Neck/backbone SSM 内核不一致

ETFHead 维度对不上

修正上述问题后，你的迁移方案就是 完全正确、能跑通、能得到论文级别性能的 NetMamba-FSCIL V1.0 版本。