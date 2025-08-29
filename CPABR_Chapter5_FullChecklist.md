

# CP-ABR++ Chapter 5 全量修改 Checklist（不删减版）

## 0) 元要求（开篇总纲）

-  **通读并吃透**附件 `chapter5.tex` 模版；所有结构、环境、标注、引用必须**与附件一致**。
-  **统一配色**：蓝色表示 XX，红色表示 CC；风格：功能性、低饱和度、风格统一、结构清晰。
-  **加入 zero-shot LLM baseline** 段落（如 GPT-4、LLaMA-3）。
-  **保留三档产品化配置**：Full / Lite-A / Lite-B（列表呈现）。
-  **把既往讨论的技术点、质疑点、精简思路**全部串联，**按附件格式落地**。
-  说明并突出台章创新：**异质–异配图、级联能量门、探针、条件流、解释**。

------

## A) 按附件 LaTeX 模版实施（排版与风格）

### A1. 章节骨架（与附件同款）

-  `\chapter{第五章：CP-ABR++——级联、高效、鲁棒且可解释的文本OOD检测}\label{chap:chapter5}`
-  `\section{引言}\label{introduction}`
-  `\section{模型概述}\label{cpabr_overview}`
  -  `\subsection{相关定义与术语}`
  -  `\subsection{总体框架设计}`（放“直觉图”）
  -  `\subsection{总体模型}`（放“全架构大图”）
-  `\section{CP-ABR++ 各模块}\label{cpabr_modules}`
  -  `\subsubsection{Stage-0 级联能量门}\label{stage0}`
  -  `\subsubsection{Stage-1 异质—异配语义因果图}\label{stage1}`
  -  `\subsubsection{Stage-2 多样性自适应探针}\label{stage2}`
  -  `\subsubsection{Stage-3 多模态条件流检测}\label{stage3}`
  -  `\subsubsection{Stage-4 证据链解释生成}\label{stage4}`
-  `\section{实验与分析}\label{evaluation}`（数据、指标、基线、主表、消融、敏感性、案例）
-  `\section{小结}\label{conclusion}`

### A2. 图表/算法环境（照附件）

-  **图**：`\begin{figure}[htp]\centering\includegraphics[width=1\columnwidth]{figures/chap5/...}\caption{中文标题}\label{...}\end{figure}`
-  **表**：`\begin{table}[htp]\centering\scriptsize\caption{中文标题}\begin{tabular}...\end{tabular}\label{...}\end{table}`（保持附件把 `\label` 放在 `tabular` 后）
-  **算法**：`\begin{algorithm}[h]\small\caption{中文标题}\label{...}\begin{algorithmic}[1] ... \end{algorithmic}\end{algorithm}`（`\Require \Ensure \STATE` 风格）

### A3. 公式与引用

-  统一用 `equation` 带编号：`\begin{equation} ... \label{...}\end{equation}`；**不用** `equation*`。
-  引用用法：`如图~\ref{...}所示`、`如表~\ref{...}`、`式~(\ref{...})`，参考文献 `~\cite{...}`。

### A4. 标签命名

-  建议前缀：`fig:` / `tab:` / `eq:` / `alg:`（更安全）；
-  若坚持“完全照附件短标签”，也可，但**必须去重**避免重名。

------

## B) 内容落地（逐段写什么）

### B1. 5.1 引言

-  交代文本 OOD 背景：近 OOD、过度自信、流形纠缠、语义偏移 vs 协变量偏移。
-  明确“**异配性（heterophily）\**挑战”：图中相连节点常不同类引发近域混淆 → 传统同配 GNN 退化 → 采用\**异质—异配图**缓解流形纠缠。
-  方法动机：**级联筛选（效率） + 语义因果图（结构化语义/抗近OOD） + 探针（主动扩边） + 条件流（可靠打分） + 证据链解释（可信）**。
-  **贡献点**：精炼 4–5 条。

### B2. 5.2 模型概述

-  定义术语：ID/OOD、Near/Far-OOD、能量分数 E(x)E(x)、异常分数 s(x)s(x)、语义因果图 G=(V,E,W)G=(V,E,W)、**异配率**。
-  **直觉图（图5.1）**：t-SNE 风格对比（传统 vs CP-ABR++），中文图题与说明。
-  **总体模型图（图5.2）**：分块展示 Stage-0…4 数据流，排版对齐附件。

### B3. 5.3 模块与关键公式

-  **Stage-0 能量门**
  -  式：`E(x) = -T\log \sum_k e^{f_k(x)/T}`（式(5.x)）
  -  阈值 τE\tau_E 设定与早停动机。
-  **Stage-1 异质—异配语义因果图**
  -  多类型节点/边（异质）；相连节点倾向不同类（异配）。
  -  **异配友好聚合**（任选一种并写简洁公式：高阶邻域/反同配权重/ppr/线性滤波等）。
  -  **LLM 构图 + 幻觉自检**（置信阈值/一致性问答/降权/删边）→ 图编码得 h∗\mathbf{h}_*。
-  **Stage-2 多样性自适应探针**
  -  识别不确定区域（高熵/最小马氏距/能量边缘）。
  -  生成 {xprobe}\{x_{\text{probe}}\}（条件扩散或 GAN）。
  -  **损失**：对抗 + 多样性（两式简述）。
-  **Stage-3 多模态条件流**
  -  CNF：`\log p_X(x|\mathbf{h}_*) = \log p_Z(f_\theta(x|\mathbf{h}_*)) + \log|\det J|`
  -  先验：GMM；**排斥损失** Lrepel\mathcal{L}_{\text{repel}}；
  -  异常分数：s(x)=−log⁡pX(x∣h∗)s(x) = -\log p_X(x|\mathbf{h}_*)。
-  **Stage-4 证据链解释**
  -  图证据（连通性/删边/簇间断裂）+ 流证据（潜空间离群/最近成分马氏距）。
  -  模板驱动 LLM 生成自然语言解释。
-  **总损失与伪码**
  -  L=LID-cls+αLflow+βLrepel\mathcal{L}=\mathcal{L}_{\text{ID-cls}}+\alpha\mathcal{L}_{\text{flow}}+\beta\mathcal{L}_{\text{repel}}（编号入文）。
  -  **算法5.1**：训练/推理步骤（带行号、用附件算法环境）。
-  **每小节末标 Lite 路线**
  -  **Lite-A**：S0 + S1 + S3（无探针）
  -  **Lite-B**：S0 + S3（最小可行）
  -  说明不影响核心叙事，呼应奥卡姆剃刀质疑。

### B4. 5.4 实验与分析

-  **数据集**：CLINC150、Banking77（留出近OOD）、ROSTD（零样本）、ToxiGen（对抗远OOD）；**表5.1** 统计（`booktabs` + `\scriptsize`）。
-  **指标**：AUROC、AUPR、FPR@95TPR、Open-set Acc、ID Acc（中文定义）。
-  **实现细节**：PyTorch；编码器二选一（BERT 或 LLaMA，全文保持一致）；GNN层数、K、训练时长、硬件（A100）。
-  **基线**：MSP、ODIN、Mahalanobis、Energy、VI-OOD、CED…（一句话+引用）。
-  **主结果表（表5.2）**：CLINC150 / Banking77 两块，列 AUROC/AUPR/FPR@95/ID Acc。
  -  若含“模拟/预期”数值，**表注星标**（如 *），正文不过度渲染。
-  **消融表（表5.3）**：去 S1/S2/S3（或 S3 改能量分），以及“单高斯 vs GMM”。
-  **超参敏感性图（图5.3）**：GNN 层数、探针数 MM 对 AUROC；用附件图环境与字号。
-  **可视化/案例（图5.4+短文）**：t-SNE（基线 vs Ours）+ 1–2 个近 OOD 案例。
-  **效率分析**：级联门过滤率、平均 ms/样本，与复杂基线对比一句话“性价比”。

### B5. 5.5 小结

-  3–5 句中文小结：强调 **异质—异配图**解决近 OOD、**探针**主动扩边、**条件流**可靠打分、**解释**可审计；
-  未来工作：轻量化构图/多语种/在线持续学习。

------

## C) 术语与记号统一

-  分数记号只用：能量 E(x)E(x) 与异常 s(x)s(x)。
-  Near/Far-OOD 中英统一（Near-OOD / Far-OOD）。
-  图名统一：**异质—异配语义因果图**（Heterogeneous-Heterophilic Semantic-Causal Graph），首次给脚注/括注。
-  阶段名称、表/图标题统一写法；变量体例：向量粗体、函数正体、集合黑板体（与附件一致）。

------

## D) 预防式回应（质疑点）

-  **奥卡姆剃刀**：引言或概述注明 Full 与 Lite 并举；差距由消融证明**非堆复杂度**。
-  **端到端**：Enc+GNN+Flow 可微联合优化；LLM 构图/解释为**外部辅助（不可微）**。
-  **数据泄露/幻觉**：Stage-1 交代自检与降权/删边策略。
-  **模拟数值**：统一星标，强调**相对改善**与**趋势**。

------

## E) 工程资源与文件结构

-  **图片**放 `figures/chap5/`，文件名：`intuitive_space.pdf`（图5.1）、`architecture.pdf`（图5.2）、`sensitivity.pdf`（图5.3）、`tsne_compare.pdf`（图5.4）。
-  **表标签**：`tab:datasets`，`tab:main`，`tab:ablation`。
-  **公式标签**：`eq:energy`，`eq:cnf`，`eq:repel`，`eq:loss_total`。
-  **算法标签**：`alg:cpabr`。
-  若沿用附件短标签，**务必自动去重**。

------

## F) 精简路线（不降创新、尽量保性能）

-  **Lite-A**（默认推荐）：S0 + S1（异质—异配图）+ S3（CNF/GMM），去 S2。
-  **Lite-B**（最小可行）：S0 + S3；S1 融为检索型邻域条件（不跑 GNN，仅 KNN 统计条件），**说明性能 trade-off**。

------

## G) 两个决策点（需定稿前敲定）

-  **标签风格**：是否采用 `fig/tab/eq/alg` 前缀？（建议采用，**更安全**）
-  **编码器基座**：全文统一选 **BERT-base** 或 **LLaMA-2-7B**（二选一，避免冲突）。

### G-1. 编码器基座选择推荐（保留说明与结论）

-  **推荐主用 BERT-base**（110M，轻量、句子级高效，契合级联/流式评分；LLaMA-2-7B 成本高）。
-  **备选**：RoBERTa-base（更鲁棒）、DistilBERT（更快）。
-  **比较策略**：A/B 报告 LLaMA-2/3-7/8B（量化可选），如 AUROC +1–2% 但延迟 +20% 的对比。
-  实施：HF 加载 BERT-base，提取 [CLS][CLS] 向量用于 Stage-1 与 S3 条件。

------

## H) “落地清单”补充（含优先级 P0/P1/P2）

### H-1. 写作与定位

-  **P0**：若无严格干预/反事实，**降级“因果”措辞**为“结构化语义/知识图（带因果假设的证据化校验）”；
-  **P1**：若保留“因果”，补 **SCM + do-operator + 反事实探针 + 指标 + 正则**；
-  **P0**：对每个 Stage 交代“独立作用/不可替代/失败模式”；给出 Full/–S2/–S1/S0+S3/E2E-lite **矩阵**；
-  **P0**：LaTeX 与版式**完全对齐附件**、符号统一、图题风格、星标模拟值。

### H-2. 方法与算法

-  **P0**：Stage-0 阈值策略（ID 验证集 95% 分位）、报告早停与延迟；
-  **P0**：Stage-1 定位为 **HHG（异质—异配）**；选用 H2GCN / GPR-GNN / MixHop / FAGCN / CPGNN 等**异配友好**；可用 R-GCN 处理异质关系；
-  **P0**：LLM 边生成 + 自检 + 结构重建/一致性正则（小权重）；
-  **P1**：Stage-2 仅训练期启用，损失 Ladv−ηLdivL_{\text{adv}}-\eta L_{\text{div}}；**轻量替代**：检索式/语义扰动/模板编辑（S2-lite）；
-  **P0**：Stage-3 CNF+GMM，最终 s(x)s(x)；可用能量分或密度比作极简替代并在消融报告退化；
-  **P1**：Stage-4 证据来源与**遮蔽/删除忠实性检验**（sanity check）。

### H-3. 复杂度与工程

-  **P0**：训练范式：“可微核心（Enc/GNN/Flow）+ 外部辅助（LLM 构图/解释）”；
-  **P0**：总损失 Ltotal=Lce+αLflow+βLrepel+γLinv\mathcal{L}_{\text{total}}=L_{\text{ce}}+\alpha L_{\text{flow}}+\beta L_{\text{repel}}+\gamma L_{\text{inv}}（若用因果正则）；
-  **P0**：效率报告：均值/尾延迟、S0 截断率、显存；给出 S0+S3 / S0+S1 **轻量路径**；
-  **P0**：Backbone 对照表；Zero-shot LLM baseline；
-  **P0**：可复现清单：随机种子、超参表、阈值、prompt 模板、划分脚本、代码链接。

### H-4. 实验协议

-  **P0**：数据划分一致、预处理与 token 长度统一、同一评测脚本；
-  **P0**：指标：AUROC/AUPR/FPR@95/Open-set/ID Acc + 效率（ms/样本、VRAM）；
-  **P0**：消融：–S1 / –S2 / S3→Energy / 单高斯；超参：L、k、M、K、λ,η\lambda,\eta；**帕累托曲线**；
-  **P1**：可视化与案例（近-OOD）；
-  **P0**：公平性声明：温度、上下文、模板、LLM 版本日期、同一 ID 训练与流程。

### H-5. 风险与边界

-  **P1**：LLM 幻觉与偏差：自检+边预算+人工抽检样本；
-  **P1**：泛化边界：领域漂移/多语；
-  **P1**：安全：用 ToxiGen 做对抗测试与拒识策略讨论。

### H-6. 里程碑（先后顺序）

-  **P0**：术语/因果措辞调整或补齐因果要素；
-  **P0**：HHG 与异配友好 GNN 落地；
-  **P0**：Backbone & Zero-shot 主结果两张表；
-  **P0**：关键消融 + 效率；
-  **P0**：LaTeX 标签/风格与附件对齐；
-  **P1**：反事实探针与 Ij/LinvI_j/L_{\text{inv}} 小实证；
-  **P1**：解释忠实性 sanity check；
-  **P1**：可视化与案例；
-  **P2**：更多数据集/更大基座；
-  **P2**：在线/持续学习、多语言扩展。

------

## I) 5.X 可插拔与产品化配置（整段 LaTeX 需插入）

-  在 5.3 或 5.4 之间新增小节：**“可插拔设计与产品化配置”**；
-  插入下列 **完整 LaTeX 模板**（需 `\usepackage{booktabs}`；`tab:ablation` 替换为实际 label；“Δ”列填入你的消融差值）：

```latex
\subsection{可插拔设计与产品化配置}
\label{subsec:plug-config}

\paragraph{可插拔模块.}
CP-ABR++ 各阶段可独立开关以适配不同的部署约束：
Stage-0（Energy-Gate, 速裁）；
Stage-1（异质—异配语义图 \& GNN, 结构化增强）；
Stage-2（自适应/反事实探针, 训练期增强）；
Stage-3（条件流 + 多模态先验, 可靠评分）；
Stage-4（证据链解释, 可解释性）。
训练期与推理期的可用性区别：S2 仅在训练期启用，S0/S1/S3/S4 可在推理期按需启用。

\begin{table*}[t]
\centering
\caption{CP-ABR++ 三档产品化配置（可插拔）。``训练开销/推理开销''相对 Full 归一化；``预期变化''建议用消融结果填充。}
\label{tab:product-config}
\begin{tabular}{@{}l l l l l l@{}}
\toprule
配置 & 模块组合（推理期） & 模块组合（训练期） & 训练开销 & 推理开销 & 预期性能变化（vs Full） \\
\midrule
\textbf{Full} &
S0 + S1 + S3 + S4 &
S0 + S1 + \underline{S2} + S3 &
1.0 & 1.0 &
基准（最佳 AUROC/FPR@95；有解释） \\
\textbf{Lite-A} &
S0 + S1 + S3 &
S0 + S1 + S3 &
$\approx$0.7–0.8 & $\approx$0.8 &
AUROC \,$-\Delta_{\small\text{S2}}$；FPR@95 \,$+\Delta_{\small\text{S2}}$；无解释 \\
\textbf{Lite-B} &
S0 + S3 &
S0 + S3 &
$\approx$0.5–0.6 & $\approx$0.6 &
AUROC \,$-\Delta_{\small\text{S1+S2}}$；FPR@95 \,$+\Delta_{\small\text{S1+S2}}$；无图/无解释 \\
\bottomrule
\end{tabular}
\end{table*}

\paragraph{适用场景建议.}
\begin{itemize}
  \item \textbf{Full}：近-OOD 压力大、需审计/合规解释或高风险业务；延迟预算中高。
  \item \textbf{Lite-A}：对可解释性要求一般，但需稳健 near-OOD 检测；中等算力/延迟预算。
  \item \textbf{Lite-B}：强实时/边缘端；容忍少量性能折损，追求极低延迟与工程简化。
\end{itemize}

\paragraph{配置—消融对应关系.}
Full 对应完整模型；Lite-A \(\approx\) “\textit{–S2}” 消融（见表~\ref{tab:ablation}）；Lite-B \(\approx\) “\textit{–S1, –S2}” 消融。将表~\ref{tab:ablation} 的数值差异填入表~\ref{tab:product-config} 的“预期变化”列，可量化取舍。

\paragraph{开关与参数.}
默认开关：\texttt{use\_graph=true}, \texttt{use\_probe=true}, \texttt{use\_explain=true}, \texttt{scorer=cflow}.\\
Full: 全开；Lite-A: \texttt{use\_probe=false}, \texttt{use\_explain=false}; Lite-B: \texttt{use\_graph=false}, \texttt{use\_probe=false}, \texttt{use\_explain=false}.\\
在 \(\)S0 中采用 ID 验证集 95\% 分位阈值 \(\tau_E\)；S3 采用条件流 + GMM 先验（Lite-B 以编码器表征直连 S3）。

\paragraph{运行时选择策略（可选）.}
当 \(E(x) < \tau_E^\text{low}\) 直接判 ID；当 \(E(x) > \tau_E^\text{high}\) 直接判 OOD；
其余样本按配置进入 S1{+}S3（Full/Lite-A）或直接 S3（Lite-B）。该双阈值策略可进一步降低平均延迟。
```

------

## J) “因果最小闭环”融合（保留因果/或降级说明）

### J-1. 术语与标题（若保留“因果”）

-  统一用“**可干预语义图（causal-inspired semantic graph）**”；
-  引言/Stage-1 开头加**澄清**语句：通过概念开关干预与反事实探针估计**干预敏感度**，为证据化校验，**不主张严格可识别性**。
-  Stage-1 小节名可写：**“可干预语义图构建与一致性自检（HHG）”**；图题明确“**异质—异配语义图（HHG）**”。

### J-2. 极简 SCM 与变量定义（插入 5.2）

-  插入链式 SCM（方程需编号）：
  -  C→X→Z→Y^C \rightarrow X \rightarrow Z \rightarrow \hat{Y}；
  -  C={Cj}C=\{C_j\} 为概念开关；干预 do(Cj=1/0)do(C_j{=}1/0)。

### J-3. Stage-1 异配友好消息传递（插入公式）

-  插入并解释（含异配信号 κij\kappa_{ij}）：

  - [ ]

  hi(ℓ+1)=σ ⁣(W0(ℓ)hi(ℓ)+∑r∈R∑j∈Nirαij(r) Wr(ℓ)hj(ℓ)), αij(r)∝exp⁡ ⁣(ϕr(i,j)⋅κij)\mathbf{h}_i^{(\ell+1)}=\sigma\!\left( W_0^{(\ell)}\mathbf{h}_i^{(\ell)} + \sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_i^r} \alpha_{ij}^{(r)}\, W_r^{(\ell)} \mathbf{h}_j^{(\ell)} \right),\  \alpha_{ij}^{(r)} \propto \exp\!\big(\phi_r(i,j)\cdot \kappa_{ij}\big)

  -  交代 κij∈{+1,−1}\kappa_{ij}\in\{+1,-1\} 的来源与作用。

### J-4. Stage-2 反事实/干预探针 + 一致性损失

-  生成成对 (x,xjcf)(x, x^{cf}_j)（doCjC_j 插入/移除/替换）；
-  加入 Lcf=1∣P∣∑(x,xjcf)∣s(x)−s(xjcf)−Δj∣\mathcal{L}_{\text{cf}} = \frac{1}{|\mathcal{P}|}\sum_{(x, x^{cf}_j)}\big| s(x)-s(x^{cf}_j) - \Delta_j \big|。

### J-5. Stage-3 因果敏感度与稳定性正则

-  定义 Ij=E[s(x)∣do(Cj=1)]−E[s(x)∣do(Cj=0)]I_j=\mathbb{E}[s(x)\mid do(C_j{=}1)] - \mathbb{E}[s(x)\mid do(C_j{=}0)]；
-  Linv=∑j∣Ij∣\mathcal{L}_{\text{inv}}=\sum_j |I_j|；
-  总损失更新：Ltotal=LID-cls+αLflow+βLrepel+γLcf+ζLinv\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{ID-cls}}+\alpha \mathcal{L}_{\text{flow}}+\beta \mathcal{L}_{\text{repel}}+\gamma \mathcal{L}_{\text{cf}}+\zeta \mathcal{L}_{\text{inv}}（γ,ζ\gamma,\zeta 小权重）。

### J-6. 伪代码微调（两行）

-  `GenerateCounterfactualProbes via do(C_j)`
-  `Estimate I_j and update L_inv`

### J-7. 小型消融（支撑因果）

-  增加“– 去干预与 LinvL_{\text{inv}}”一行：报告 AUROC↓ ~0.6–1.2、FPR@95↑ ~2–4（可为预期或实测）。

### J-8. 引言与贡献段微调

-  引言补一句“**近-OOD 的‘异配’本质**与 **HHG+概念干预**提升可分性”；
-  贡献点补：**“提出 HHG+干预+敏感度正则的因果-启发闭环，在不增推理复杂度下降低近-OOD 误报。”**

------

## K) Zero-shot LLM Baseline（必要性与选型）

-  **纳入 zero-shot LLM baseline**，符合 24–25 年趋势与审稿预期。
-  **模型**：优先 GPT-4（或 GPT-4o）；若强调开源，用 **LLaMA-3-8B-Instruct**；可备 GPT-3.5 / Mistral-7B。
-  **设置**：统一 prompt、温度=0；在 CLINC150 / Banking77 / ROSTD 报告 AUROC / FPR@95。
-  **对照**：展示我们在 Near-OOD 上更低 FPR@95 与效率优势；
-  **表格**：可在主表或附录增加 “BERT-base vs LLaMA-2/3-8B（量化） vs Zero-shot GPT-4/LLaMA-3” 的**性能-延迟**对比。
-  **趋势注记**：zero-shot 常优于传统后处理，弱于调优/结构化方法 → 突出 CP-ABR++ 的系统性优势。

------

## L) 路径 B（如需保留“因果”招牌的最小要素）

-  **SCM 骨架**：C ⁣→ ⁣X ⁣→ ⁣Z ⁣→ ⁣Y^C\!\rightarrow\!X\!\rightarrow\!Z\!\rightarrow\!\hat{Y}；
-  **可执行干预**：通过可控编辑/生成实现 do(Cj=1/0)do(C_j{=}1/0) 与反事实探针；
-  **可量化指标**：IjI_j 与 Linv\mathcal{L}_{\text{inv}}；可用于边权再标定或稳定性正则；
-  **文稿同步**：Stage-1 更名为“因果-可干预语义图”；新增“无干预 vs 有干预/不变性正则”的对比行并解释提升原因。

------

### ✅ 收尾提醒

-  决定**是否采用标签前缀**（建议：采用）。
-  决定**统一编码器基座**（建议：BERT-base 为主，附 LLaMA-2/3 变体对照）。
-  在**主结果表**中纳入 **Zero-shot LLM** 与 **Backbone 对照**。
-  **所有模拟/预期值**加星标并在脚注说明。
-  **附上**随机种子、超参、阈值、prompt 模板、划分脚本与代码链接。

------

