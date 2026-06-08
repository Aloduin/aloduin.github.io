---
title: 近5年EEG预训练模型综述
date: 2026-06-08 23:38:56
tags:
---
# 脑电预训练模型：从 BENDR 到 NeuroLM 的五年演进

> **摘要**：本文系统梳理 2021—2026 年间脑电（EEG）预训练基础模型的研究进展，覆盖 20 余篇代表性论文，从预训练方法论、多通道处理策略、规模化趋势到应用落地，勾勒出这一新兴领域的技术演进脉络与未来方向。

---

## 目录

1. [为什么要做脑电预训练模型？](#1-为什么要做脑电预训练模型)
2. [早期探索（2021—2022）：可行性验证](#2-早期探索20212022可行性验证)
3. [基础模型崛起（2023—2024）：范式确立](#3-基础模型崛起20232024范式确立)
4. [规模化与多模态（2025—2026）：走向大模型](#4-规模化与多模态20252026走向大模型)
5. [核心技术：如何处理多通道 EEG？](#5-核心技术如何处理多通道-eeg)
6. [预训练策略对比](#6-预训练策略对比)
7. [下游应用全景](#7-下游应用全景)
8. [主要挑战与未来方向](#8-主要挑战与未来方向)
9. [参考文献](#9-参考文献)

---

## 1. 为什么要做脑电预训练模型？

脑电图（EEG）是神经科学与临床医学领域最重要的非侵入性信号之一。凭借毫秒级时间分辨率、低成本与便携性，EEG 在癫痫诊断、睡眠分析、脑机接口（BCI）和认知神经科学中具有不可替代的价值。

然而，传统 EEG 深度学习模型面临根本性困境：

- **标注成本极高**：EEG 标注需要专业领域知识，高质量标注数据极度匮乏
- **泛化能力差**：模型通常针对特定任务和数据集设计，换个场景就失效
- **异构性突出**：不同设备的电极数量（19 到 256 不等）、导联配置、采样频率差异显著
- **个体差异大**：同一任务下不同被试的 EEG 信号分布差异极大，"跨受试者泛化"至今仍是悬而未决的难题

受 NLP 领域 BERT/GPT 和 CV 领域 MAE/ViT 基础模型成功的启发，研究者们自 2021 年起开始系统性地探索将**大规模自监督预训练**引入 EEG 领域——用海量无标签 EEG 数据预训练通用表征，再以少量标注数据微调适配下游任务。

这一思路的核心洞见在于：**神经振荡模式是可迁移的通用知识**。无论是癫痫的棘慢波、睡眠的纺锤波，还是运动想象的 mu 节律，这些规律性特征都蕴含在未标注的 EEG 信号中，等待模型去发现。

---

## 2. 早期探索（2021—2022）：可行性验证

### BENDR（2021）——开创性的第一步

> *Kostas et al., Frontiers in Human Neuroscience, 2021*

BENDR 是 EEG 基础模型领域公认的先驱工作。研究者们观察到，EEG 信号与语音信号在结构上存在高度相似性——两者都是时序信号，都具有局部相关性和全局依赖性。于是他们将语音预训练领域的 Wav2Vec 2.0 框架迁移至 EEG：

**核心设计：**

```
原始 EEG 信号
    ↓ 1D CNN（局部特征提取）
    ↓ 量化（连续特征 → 离散码字）
    ↓ Transformer（上下文建模）
    ↓ 对比学习目标：最大化真实量化码与预测结果之间的互信息
```

BENDR 最重要的贡献不在于技术细节，而在于**证明了可行性**：大规模无标签 EEG 数据确实可以用来学习有意义的通用表征。这为整个领域打开了一扇门。

**局限：** 模型规模较小，仅支持固定导联配置，无显式空间位置编码。

---

## 3. 基础模型崛起（2023—2024）：范式确立

这一阶段是 EEG 基础模型的"爆发期"，多个代表性方法密集涌现，核心范式逐步确立。

### BIOT（NeurIPS 2023）——跨模态统一编码器

> *Yang et al., NeurIPS 2023*

BIOT 的核心问题意识是：**如何让一个模型同时处理 EEG、ECG、EMG 等不同模态的生物信号，并应对不同数据集之间的通道数差异？**

解决方案简单却高效：**将每个通道的时间段独立切分后"排列成句子"**。

```
设备 A：32 通道 × T 时间段 → 32T 个 token
设备 B：19 通道 × T 时间段 → 19T 个 token
                ↓ 统一 Transformer 处理
```

不同通道数的数据自然地被统一为不同长度的"句子"，模型无需修改即可处理。预训练采用类 BYOL 的对比学习策略：动量编码器生成目标表征，在线编码器学习对齐。

**亮点：** 在 CHB-MIT 癫痫检测数据集上相比此前 SOTA 提升约 4%，并在 ECG 分类、人体活动识别等任务上同样有效——首次验证了跨模态生物信号统一预训练的价值。

---

### Brant（NeurIPS 2023）——颅内信号的基础模型

> *Zhang et al., NeurIPS 2023*

与头皮 EEG 不同，颅内神经信号（iEEG、sEEG、LFP）直接记录神经元活动，信噪比更高，空间分辨率更好，但获取难度极大。Brant 是**首个专门面向颅内神经信号的大型基础模型**，采用掩码重建与对比学习联合预训练，并针对颅内电极的空间拓扑结构专门设计了位置编码。

---

### LaBraM（ICLR 2024 Spotlight）——里程碑式工作

> *Jiang et al., ICLR 2024*

LaBraM（Large Brain Model）是 EEG 基础模型领域目前最具影响力的工作之一，其核心创新是**向量量化神经频谱预测（VQ-NSP）**。

**两阶段预训练框架：**

**第一阶段：训练神经分词器**

```
EEG 片段 (1 通道 × T 时间点)
    ↓ CNN 编码器（局部频谱特征提取）
    ↓ VQ-VAE（量化为码本中的离散 code_id）
    ↓ 重建目标：频谱振幅 + 相位
```

**第二阶段：训练神经 Transformer**

```
随机掩码约 75% 的 EEG 通道片段
    ↓ 神经 Transformer（Bidirectional）
    ↓ 预测被掩码片段的 code_id（而非原始波形！）
```

这一设计有深刻的动机：直接重建原始 EEG 波形意味着模型要花大量容量去拟合噪声；而离散 code_id 对应频谱上有语义意义的模式，预测难度适中，且形式上与 NLP 中的 BERT（预测词表 token）完全统一——为后续与 LLM 融合埋下伏笔。

**关键设计细节：**

| 设计要素 | 具体方案 |
|---------|---------|
| 通道处理 | 每通道独立作为一个 token 序列 |
| 通道位置编码 | 基于 10-20 系统的标准电极索引（跨数据集同名电极共享嵌入） |
| 时间位置编码 | 标准 1D 可学习位置嵌入 |
| 支持导联范围 | 19—64 导联任意配置 |
| 参数规模 | 5.8M（Base）/ 46M（Large）/ 369M（Huge）|
| 预训练数据 | ~2500 小时，约 20 个公开数据集 |

**主要结果：**

- TUAB（Temple University Abnormal EEG）异常检测：**90.3% 准确率**
- SEED/DEAP 情绪识别：超越全监督基线
- 步态预测、事件分类：多任务 SOTA

LaBraM 成功应对了 EEG 基础模型的三大核心工程挑战：电极数量不一致、样本长度不等、数据集格式差异大。

---

### NeuroGPT（2024）——大规模 GPT 风格预训练

> *Cui et al., 2024*

NeuroGPT 借鉴 GPT 架构，在 **5656 小时**的大规模 EEG 数据上进行自回归预训练，参数量 79.53M。虽然在技术创新上不如 LaBraM 激进，但**开源权重和超大规模数据**对社区研究贡献显著。

---

### EEGPT 双雄（2024）——双重自监督 vs 缩放律

同年出现了两篇同名 EEGPT 论文，各有侧重：

**EEGPT（Wang 等，NeurIPS 2025）**——提出双重自监督框架：
- 损失 1：掩码 patch 重建（MSE）
- 损失 2：在线编码器与动量编码器输出对齐（类 BYOL）

两个目标的结合在训练稳定性与表征质量之间取得了平衡，提供 4.7M—25M 多个参数规模。

**EEGPT（Yue 等，2024）**——专注于**缩放律研究**，设计了 1.46M 到 **1.09B** 共 7 个参数规模档位，采用纯因果自回归预训练（next-signal prediction）。

核心发现：**在更大参数规模下，自回归预训练的迁移性能显著优于双向掩码方法**。这一结论深刻影响了后续 EEG 大模型的架构选择——EEG 领域也将走向 GPT-style 的缩放路径。

---

## 4. 规模化与多模态（2025—2026）：走向大模型

### NeuroLM（ICLR 2025）——EEG × LLM 的深度融合

> *Jiang et al., ICLR 2025*

NeuroLM 是 LaBraM 的大幅扩展，也是目前**参数量最大、数据量最多**的 EEG 基础模型：

| 指标 | LaBraM | NeuroLM |
|------|--------|---------|
| 预训练数据 | 2,500 小时 | **25,000 小时**（10×）|
| 最大参数量 | 369M | **1.7B** |
| 模型类型 | EEG Transformer | **EEG + LLM 融合** |

**三阶段预训练流程：**

```
① 神经分词器预训练
   EEG 片段 → CNN → VQ 量化 → 码本向量对齐 LLM 词嵌入空间

② LLM 多模态自回归预训练
   [EEG token₁, EEG token₂, ..., text token₁, text token₂, ...]
                ↓ LLaMA-3 等大型语言模型
           预测序列中的下一个 token

③ 对抗域对齐
   消除不同 EEG 数据集之间的分布偏移
```

关键突破在于第一步：将 EEG 的 VQ 码本向量**对齐到 LLM 的文本词嵌入空间**，使 EEG token 与文本 token 可以在同一语义空间中共存。EEG 信号从此可以作为 LLM 的"第三模态"参与自然语言推理。

**意义：** NeuroLM 不仅是规模的突破，更是范式的突破——EEG 信号从"需要专用模型处理"走向"可以用语言模型统一理解"。

---

### CBraMod（2025）——高效的时空交叉注意力

> *Wang et al., 2025*

**核心问题：** 将 C 个通道的 T 个时间步展平为 (C×T) 个 token 后，全注意力的复杂度是 O((CT)²)，对于 64 通道 × 200 时间步的 EEG，计算量约为 164 亿次操作——代价极大。

**CBraMod 的方案：Criss-Cross（十字交叉）注意力**

```
初始状态：EEG 组织为 C×T 的二维网格

奇数层：时间注意力（每个通道内，T 个 token 互相注意）
   复杂度：O(C × T²) = O(CT²)

偶数层：空间注意力（每个时间步，C 个 token 互相注意）
   复杂度：O(T × C²) = O(TC²)

总复杂度：O(CT² + TC²) = O(CT(T+C))
vs 全注意力：O(C²T²)
```

当 C=64, T=200 时，复杂度从约 1.6×10⁸ 降至约 1.7×10⁷，节省约 **90%** 的计算量。

CBraMod 在 12 个公开数据集上预训练，在运动想象（PhysioNet-MI、SHU-MI）、睡眠分期（ISRUC）、癫痫检测（CHB-MIT）、想象言语分类等 5 个任务上全面验证，取得竞争力性能。

---

### REVE 与 HEAR（2025）——彻底解决导联异构

两项工作分别从不同角度攻克了 EEG 基础模型最棘手的工程问题：**不同采集设备使用不同导联配置**，模型无法直接泛化。

**REVE** 引入**4D 位置编码**（时间 t + 通道序号 ch + 头皮 x 坐标 + 头皮 y 坐标），在 **25,000 名受试者**的超大规模数据上预训练。对于从未见过的导联配置，只需知道电极的头皮坐标，即可通过**坐标插值**生成合理的位置编码。

**HEAR** 走得更彻底：用一个**动态函数**（MLP 等）将电极的 3D 空间坐标直接映射为通道嵌入向量，彻底摆脱了"通道名/索引查表"的方式。任何新电极，只要提供坐标，就能即插即用。

---

### LUNA（2025）——拓扑无关的极致方案

LUNA 提出了从根本上解决导联不兼容的方法：**可学习查询（Learnable Queries）+ 交叉注意力**。

```
输入：任意 N 个通道的 EEG token（N 可以是 19、64、128...）
         ↓ 交叉注意力
可学习查询（固定 Q 个）
         ↓
固定维度的潜在表征（Q × d）
         ↓ 所有下游任务统一使用这个表征
```

无论输入是 19 导联还是 256 导联，输出始终是固定维度的潜在表征。下游任务完全不感知通道配置的存在——这是比 REVE 和 HEAR 更彻底的解耦。

---

### CSBrain（2025）——神经科学先验驱动的建模

> *Zhou et al., 2025-2026*

CSBrain 引入了一个其他模型鲜少关注的维度：**神经解剖学先验**。

核心设计：将电极按神经解剖学脑区（额叶、颞叶、顶叶、枕叶等）分组，在**组内（region-level）**和**组间（cross-region）**分别建模空间关系，同时使用**跨窗口**（时间尺度）和**跨脑区**（空间尺度）的双重掩码重建策略。

这一设计使模型能够学习不同脑区之间的功能连接模式，与神经科学知识更好地对齐。

---

### 其他值得关注的工作

| 模型 | 核心特点 |
|------|---------|
| **CodeBrain**（2025）| 在最大公开 EEG 语料库 TUEG（9000+小时）上预训练；解耦分词器（独立预训练）提升模块化与可解释性；多尺度架构同时建模局部细节与全局趋势 |
| **LEAD**（2025）| 首个专门面向阿尔茨海默症检测的 EEG 基础模型；多层次对比预训练捕捉 AD 神经生理标志物 |
| **BrainPro**（2025）| 引入脑状态感知一致性损失——要求同一脑状态（清醒/睡眠/认知任务）的 EEG 表征在潜空间中聚类 |
| **BrainWave**（2024）| 专注临床应用，在多中心医院真实数据上预训练，目标是临床可部署 |
| **Brant-X**（2024）| 将 Brant 扩展至 EEG/EOG/ECG/EMG 四模态对比对齐预训练 |
| **EEG Foundation Challenge**（NeurIPS 2025）| 首个大规模 EEG 基础模型竞赛，3000+ 受试者 × 6 种认知任务，推动跨受试者泛化研究规范化 |

---

## 5. 核心技术：如何处理多通道 EEG？

多通道 EEG 数据本质上是一个 **C（通道数）× T（时间点）的矩阵**。预训练模型面临的核心工程挑战是：不同数据集的 C 值差异显著（19—256），且电极的空间排布各不相同。当前主流方案可分为四类：

### 策略一：通道独立编码（主流方案）

**代表：BIOT、LaBraM、NeuroLM**

将每个通道的时间段独立切分为一个 patch/token，C 个通道 × K 个时间段 = C×K 个 token，全部拼接为一条序列送入 Transformer。

```
Fp1 ——→ [t1][t2][t3]
 C3 ——→ [t1][t2][t3]  →  [Fp1·t1][Fp1·t2]...[C3·t1]...[O1·t3]  →  Transformer
 Cz ——→ [t1][t2][t3]          每个 token 携带：时间位置编码 + 通道身份编码
 O1 ——→ [t1][t2][t3]
```

**优点：** 天然支持任意通道数；同名电极跨数据集共享嵌入（LaBraM 的关键设计）。

**缺点：** 仍依赖通道名称对齐；序列长度随通道数线性增长，计算量 O((CT)²)。

### 策略二：VQ 神经分词器

**代表：LaBraM、NeuroLM、CodeBrain**

不直接重建原始波形，而是先训练一个 VQ-VAE 将 EEG 片段离散化为码本中的 code_id，预训练目标变为预测 code_id（分类任务），而非重建信号值（回归任务）。

**为什么这样做更好？**
- 原始 EEG 信号噪声极大，重建噪声是对模型容量的浪费
- 离散 code 对应频谱上有语义意义的模式（振幅 + 相位的组合）
- 预测 code_id 与 NLP 中的 BERT 形式统一，天然支持与 LLM 对接

### 策略三：坐标感知位置编码

**代表：REVE（4D 编码）、HEAR（动态嵌入）、LUNA（3D 坐标）**

用电极在头皮上的三维坐标（基于国际 10-20 系统的球坐标）替代或增强离散通道索引。

```
REVE：每个 token = f(时间 t, 通道 ch, 头皮 x, 头皮 y)  ← 4D 位置编码
HEAR：通道嵌入 = MLP(x坐标, y坐标, z坐标)              ← 动态坐标函数
LUNA：3D坐标编码 + 可学习查询聚合                       ← 拓扑无关设计
```

**优点：** 对未见过的导联配置，可通过坐标插值生成合理编码，真正实现跨设备泛化。

### 策略四：时空交叉注意力

**代表：CBraMod（Criss-Cross）、CSBrain（层次化）**

将注意力拆分为时间维度和空间维度两个方向交替执行，既保留时空结构，又大幅降低计算复杂度。

```
CBraMod：O(CT²) [时间注意力] + O(TC²) [空间注意力]  =  O(CT(T+C))
vs 全注意力：O(C²T²)

当 C=64, T=200 时：节省 ~90% 计算量
```

---

## 6. 预训练策略对比

| 策略 | 代表模型 | 预测目标 | 核心优势 | 主要局限 |
|------|---------|---------|---------|---------|
| **对比学习** | BENDR、BIOT、LEAD | 对比向量 | 全局判别表征；无需重建 | 正负样本设计敏感；无显式局部建模 |
| **掩码信号重建（MSM）** | LaBraM（VQ）、CBraMod、CSBrain | 原始信号或 code_id | 强迫模型理解局部结构；类 BERT | 重建原始波形时噪声大；VQ 码本固定后灵活性有限 |
| **自回归预测** | EEGPT(Yue)、NeuroGPT | 下一个 patch | 缩放性能更优；与 LLM 天然对齐 | 因果约束损失双向上下文 |
| **EEG-LLM 融合** | NeuroLM | LLM next-token | 多任务统一；EEG-文本协同 | 需大量配对数据；训练开销极大 |
| **多模态对比对齐** | Brant-X | 跨模态向量 | 学习模态不变特征 | 依赖配对多模态数据 |

---

## 7. 下游应用全景

### 7.1 临床医疗

**癫痫检测与分类**是验证最充分的临床应用。BIOT 在 CHB-MIT 数据集上超越此前 SOTA 约 4%；BrainWave 专注多中心临床数据，目标是真正的临床可部署。

**睡眠分期**方面，LaBraM、EEGPT、CBraMod 等模型均在 Sleep-EDF、ISRUC 等标准数据集上实现了五阶段（Wake/N1/N2/N3/REM）的高精度自动分类。

**EEG 异常检测**是 LaBraM 的亮点任务：在 TUAB（Temple University Abnormal EEG，全球最大公开临床 EEG 数据集）上达到 **90.3% 准确率**，接近临床专家水平。

**神经退行性疾病**：LEAD 专门针对阿尔茨海默症，探索 EEG 作为早期筛查生物标志物的可行性。

### 7.2 脑机接口（BCI）

**运动想象解码**（区分左手/右手/双手/脚的运动意图）是 BCI 的经典任务，CBraMod、LaBraM、REVE 等均有较好验证结果。

**想象言语分类**（解码用户在心里"说"的词语）是 CBraMod 等 2025 年工作新开拓的方向，为言语障碍患者的辅助通信提供新思路。

**EEG 到图像解码**是 2024—2025 年的热点：将预训练 EEG 编码器与图像生成扩散模型结合，尝试重建被试观看图像的视觉内容。

### 7.3 认知状态与情感计算

**情绪识别**（SEED、DEAP 等数据集）是 LaBraM、EEGPT 等模型的重要验证任务，尤其在跨受试者设置下，预训练模型展现出显著的泛化优势。

**工作记忆与认知负荷**估计是 BrainPro 等模型重点关注的方向。

### 7.4 多模态与语言融合

NeuroLM 开创了 EEG 与自然语言深度融合的新范式，而 Thought2Text（NAACL 2025）则更进一步：直接利用 LLM 从 EEG 信号生成自然语言描述，实现开放词汇的"意念转文字"。

---

## 8. 主要挑战与未来方向

### 挑战一：跨受试者泛化壁垒

EEG 信号的个体间差异是目前最核心的未解难题。研究表明，**单纯扩大模型容量并不能自动解决域偏移**（CRCC 等工作的警示）。即使是 NeuroLM 的 1.7B 参数，在零样本跨受试者场景下仍与单受试者专用模型有显著差距。

未来方向：专门的跨受试者对齐机制（元学习、因果表征学习、个性化适配层）。

### 挑战二：数据规模瓶颈

与 NLP（万亿词元级别的 Common Crawl）和 CV（数十亿图片）相比，公开可用的 EEG 数据规模极为有限：目前最大的公开 EEG 语料库 TUEG 约为 9000 小时，而 NeuroLM 的 25000 小时已是业界顶尖，但与语言模型的预训练数据量仍差了 4—5 个数量级。

未来方向：跨机构 EEG 数据共享联盟；联邦学习（在隐私保护前提下分布式预训练）。

### 挑战三：评估标准不统一

当前各模型在不同数据集、不同评估协议下进行评估，模型间的横向比较极为困难。NeurIPS 2025 EEG Foundation Challenge 的出现是迈向规范化的重要一步，但标准化工作仍任重道远。

### 挑战四：临床可解释性缺失

EEG 基础模型的内部表征对临床医生而言是黑盒，限制了其在高风险临床决策场景的部署。**可信 AI（Trustworthy AI）**方法的集成——包括不确定性估计、反事实解释、类激活映射——是走向临床落地的必经之路。

### 未来展望

基于当前进展，以下几个方向值得重点关注：

1. **EEG 缩放律的系统研究**：参数量、数据量与下游性能之间的规律性关系尚未充分建立
2. **EEG × fMRI 跨模态联合预训练**：利用 fMRI 的高空间分辨率弥补 EEG 的空间局限
3. **神经生理约束的深度融合**：将频带信息、功能连接、脑区协同等神经科学先验系统地编入预训练目标
4. **联邦学习 + 差分隐私**：在不泄露原始脑电数据的前提下实现大规模分布式预训练
5. **实时 BCI 部署**：将基础模型压缩至可在边缘设备上实时推理的规模

---

## 9. 参考文献

1. Kostas D, Aroca-Ouellette S, Rudzicz F. **BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn from Massive Amounts of EEG Data.** *Frontiers in Human Neuroscience*, 2021.

2. Yang C, Westover M, Sun J. **BIOT: Biosignal Transformer for Cross-Data Learning in the Wild.** *NeurIPS*, 2023.

3. Zhang D, Yuan Z, Yang Y, et al. **Brant: Foundation Model for Intracranial Neural Signal.** *NeurIPS*, 2023, pp.26304–26321.

4. Cai D, Chen J, Yang Y, et al. **MBrain: A Multi-channel Self-Supervised Learning Framework for Brain Signals.** *KDD*, 2023, pp.130–141.

5. Jiang W B, et al. **Large Brain Model (LaBraM) for Learning Generic Representations with Tremendous EEG Data in BCI.** *ICLR 2024 (Spotlight)*.

6. Cui J, et al. **NeuroGPT: Unlocking the Future of EEG with GPT.** arXiv:2311.03764, 2024.

7. Wang G, Liu W, He Y, et al. **EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals.** *NeurIPS*, 2025, 37: 39249–39280.

8. Yue T, Xue S, Gao X, et al. **EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-Training.** arXiv:2410.19779, 2024.

9. Zhang D, et al. **Brant-X: Alignment-Based Pre-Training for Multi-Modal Biosignal Foundation Model.** arXiv, 2024.

10. Yuan Z, et al. **BrainWave: A Brain Signal Foundation Model for Clinical Applications.** arXiv:2402.10251, 2024.

11. Jiang W B, Wang Y, Lu B L, Li D. **NeuroLM: A Universal Multi-Task Foundation Model for Bridging the Gap between Language and EEG Signals.** *ICLR 2025*.

12. Wang J, Zhao S, Luo Z, et al. **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.** arXiv:2412.07236, 2025.

13. Wang Y, Huang N, Mammone N, et al. **LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection.** arXiv:2502.01678, 2025.

14. El Ouahidi B, et al. **REVE: Representation for EEG with Versatile Embeddings.** arXiv:2510.21585, 2025.

15. Ma J, Wu F, Lin Q, et al. **CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model.** arXiv, 2025.

16. BrainPro Research Team. **BrainPro: Towards Large-Scale Brain State-Aware EEG Representation Learning.** arXiv:2509.22050, 2025.

17. Zhou J, et al. **CSBrain: A Cross-Scale Spatiotemporal Brain Foundation Model for EEG Decoding.** arXiv, 2025–2026.

18. Döner T, et al. **LUNA: A Foundation Model for EEG with Topology-Agnostic Attention.** arXiv, 2025.

19. HEAR Research Team. **HEAR: An EEG Foundation Model with Heterogeneous Electrode Adaptive Representation.** arXiv:2510.12515, 2025.

20. NeurIPS 2025 Competition Track. **EEG Foundation Challenge 2025: From Cross-Task to Cross-Subject EEG Decoding.** arXiv:2506.19141, 2025.

21. Kuruppu S, et al. **EEG Foundation Models: A Critical Review of Current Progress and Future Directions.** arXiv:2507.11783, 2025.

22. Xiong W, et al. **Bridging Brain with Foundation Models through Self-Supervised Learning.** arXiv:2506.16009, 2025.

23. Lai J, et al. **A Simple Review of EEG Foundation Models: Datasets, Advancements and Future Perspectives.** arXiv:2504.20069, 2025.

---

*本文整理自 2026 年 6 月的系统性文献检索，涵盖 2021—2026 年间脑电预训练模型领域的主要进展。如有疏漏，欢迎指正。*