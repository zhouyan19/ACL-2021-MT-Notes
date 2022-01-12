# ACL 2021 Notes (MT related)

<div align="right">
	<font style="font-size:20px;">周䶮 2021.8.16~</font>
</div>

[TOC]

## 1. Mention Flags (MF): Constraining Transformer-based Text Generators

- 作者来源：Macquarie University(Sydney)，Oracle
- Link ：[Mention Flags (MF): Constraining Transformer-based Text Generators - ACL Anthology](https://aclanthology.org/2021.acl-long.9/)

- 摘要：在基于 Seq2Seq 的**输入受限句子生成**任务上，使用 Mention Flag (**MF**) 。MF 可以保证给定词语都被覆盖，更好地满足限制条件，比baseline模型和其他生成算法效果更好，达到了state of the art，并且在 lower-source setting 上表现良好。

- MF 示意：

  <img src="https://img.wzf2000.top/image/2021/08/16/image-20210816211132744.png" alt="image-20210816211132744"  />

  <center>白色和橙色圆圈即为MF</center>

- Experiments on 3 benchmarks :  

  - Common-sense Generative Reasoning (CommonGen), with keyword constraints,
  - End-to-End restaurants dialog (E2ENLG) (Dusek et al. ˇ , 2020) with key-value constraints
  - Novel Object Captioning at scale (nocaps) (Agrawal et al., 2019)

- 结论：使用 MF 的模型在受限生成任务下能达到三个Goal：(a) 生成高质量文本 (b) 较高程度地满足限制条件 (c) 生成效率足够。在上述三个 benchmark 上都达到了 SOTA 。

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818115321956.png" alt="image-20210818115321956" style="zoom: 67%;" />



## 2. Contrastive Learning for Many-to-many Multilingual Neural Machine Translation

- 作者单位：字节跳动
- Link ：[Contrastive Learning for Many-to-many Multilingual Neural Machine Translation - ACL Anthology](https://aclanthology.org/2021.acl-long.21/)

- 摘要：通过 Contrastive Learning（对比学习），辅以对齐增强方法，建立一个着重提升非英语语言翻译质量的**多语言的机器翻译系统**。文章提出了 **mRASP2**，将单语语料和双语语料囊括在统一的训练框架之下，缩小不同语言下的表达的 gap 。mRASP2在有监督、无监督、零资源的场景下均取得翻译效果的提升。

- mRASP2示意：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818112552646.png" alt="image-20210818112552646" style="zoom: 50%;" />

  取一对平行的句子（或生成的伪对），计算交叉熵损失（橙色），计算正例、负例的对比损失（黑色）

- 词对齐数据增强方法：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818114835909.png" alt="image-20210818114835909" style="zoom: 50%;" />

- 方法：模型基于 Multilingual Transformer，在每个句子前端加上一个 language identification token ；在 Encoder 端的顶部加入对比学习任务（基于假设：不同语言中同义句的编码后的表示应当在高维空间的相邻位置），在交叉熵损失的基础上加上对比损失（**C**on**TR**astive loss）；再引入词对齐数据增强方法。

- 结论：mRASP2在有监督、无监督、零资源的场景下均取得翻译效果的提升。有监督场景平均提升 1+ BLEU（接近并超过SOTA），无监督场景平均提升 14+ BLEU，零资源场景平均提升 10+ BLEU 。

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818113247224.png" alt="image-20210818113247224" style="zoom:50%;" />

  对比学习并不会降低有监督场景的性能，并且对零资源场景下的翻译性能有重大提升；单语语料的引入对所有场景的翻译性能都有明显提升。对比学习能更好地拉近多语言表示对齐语义空间。



## 3. Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation

- 作者单位：University of Zurich ，University of Edinburgh

- Link ：[Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation - ACL Anthology](https://aclanthology.org/2021.acl-long.22/)

- 摘要：当前的NMT会出现翻译结果太短、频繁出现的词过度生成等bias，也容易受训练数据噪声影响大，而这些缺点可能来自 beam search 。本文尝试用最小贝叶斯风险编码 **M**inimum **B**ayes **R**isk decoding 的方式来处理 beam search 出现的 bias、failure。实验发现，MBR 仍会出现 length 和 token frequency 的 bias，但 "increases robustness against copy noise in the training data and domain shift" 。

- MBR："the goal of MBR is to find not the most probable translation, but the one that minimizes the expected risk for a given loss function and the true posterior distribution"

- Models : standard Transformer (except that some settings)

- 结论：采用 MBR decoding 后，仍会有short translations 的问题和 token probability 的 bias 。但相比 beam search ，MBR 在面对训练数据的噪声上有更好的鲁棒性。总结来说，MBR 的表现没有总体超过 beam search ，但其鲁棒性仍让它成为一种有前景的 MAP decoding 的替代。

  

## 4. Multi-Head Highly Parallelized LSTM Decoder for Neural Machine Translation

- Link ：[Multi-Head Highly Parallelized LSTM Decoder for Neural Machine Translation - ACL Anthology](https://aclanthology.org/2021.acl-long.23/)

- 摘要：在 NMT 任务上，self-attention 的计算复杂度是 $O(n^2)$ ，而使用 LSTM 的话则需要很长的训练时间。为了使得 LSTM 可在 sequence-level 上高度并行化，本文设计了 **M**ulti-head 的 **H**ighly **P**arallelized **LSTM** , 训练比 self-attention network 稍快，比标准 LSTM 快很多，并达到了显著的 BLEU 分数提高。

- HPLSTM：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818153530838.png" alt="image-20210818153530838" style="zoom:50%;" />

  用一个 bag-of-words representation $s^t$​​ of preceding tokens 来计算 gates 、hidden states 。其中：
  $$
  s^t = \sum_{k=1}^{t-1}i^k
  $$

- Multi-head HPLSTM:

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818175711351.png" alt="image-20210818175711351" style="zoom: 40%;" />

  像 Transformer 中的 multi-head attention 一般，将几个小的 HPLSTM 并行，形成 MHPLSTM ，以限制参数规模

- Experiments：用 MHPLSTM 替换 Transformer decoder 中的 self-attention 层，在 WMT14  English to German and English to French news translation tasks 上进行实验。

- 部分结果：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818180156606.png" alt="image-20210818180156606" style="zoom:50%;" />

- 结论：MHPLSTM 比 self-attention networks 的效果更好，并且训练要 slightly faster ，同时比 self-attention Transformer decoder 在 decoding 上快得多。



## 5. A Bidirectional Transformer Based Alignment Model for Unsupervised Word Alignment

- Link ：[A Bidirectional Transformer Based Alignment Model for Unsupervised Word Alignment](https://aclanthology.org/2021.acl-long.24.pdf)

- 摘要：本文展示了 **B**idirectional **T**ransformer **B**ased **A**lignment （**BTBA**）在 word-alignment 任务中的非监督学习的运用。BTBA 关注源文本、目标文本的左右两侧，来得出精确的 target-to-source attention（alignment），预测当前的 target word 。此外，还使用 full context based optimization method 与 self-supervised training 来对 BTBA 模型进行微调。在 3 word alignment 的任务上，BTBA 比以往的 neural word alignment 方法和统计方法 GIZA++ 方法都有更好的表现。

- BTBA：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818194136747.png" alt="image-20210818194136747" style="zoom:50%;" />

- 部分实验结果：

  <img src="https://img.wzf2000.top/image/2021/08/18/image-20210818194543277.png" alt="image-20210818194543277" style="zoom: 40%;" />



## 6. Learning Language Specific Sub-network for Multilingual Machine Translation

- Link ：[Learning Language Specific Sub-netswork for Multilingual Machine Translation](https://aclanthology.org/2021.acl-long.25.pdf)

- Authors：Zehui Lin , Liwei Wu , Mingxuan Wang, Lei Li ByteDance AI Lab

- 摘要：以往的 Multilingual neural machine translation 容易趋于“平均化”，在 rich-source 的语言对上容易出现性能下降。而本文设计的 **La**nguage **S**pecific **S**ub-network (**LaSS**) 可以减少参数的影响。在 36 种语言上，LaSS 将 BLEU 平均提高了 1.2；在 new language pairs 与 zero-shot 的翻译上更是能把 BLEU 平均提高 8.3 ，其中 zero-shot 比 baseline 的 BLEU 增加了惊人的 26.5。

- LaSS：

  ![image-20210820105936880](https://img.wzf2000.top/image/2021/08/20/image-20210820105936880.png)

  <center>既有共享的权重，又有独立的权重</center>

- 方法：

  1. 先以 mTransformer 作为网络的支架，在一个 multilingual dataset $\{\mathcal{D}_{s_i \rightarrow t_i} \}_{i=1}^{N}$​ 上进行训练 （ $s_i$ 表示 source， $t_i$ 表示 target）。采用的 Loss :
     $$
     \mathcal{L} = \sum_i \sum_{\langle x,y \rangle \sim \mathcal{D}_{s_i \rightarrow t_i}} -\log P_{\theta}(\mathrm{y}|\mathrm{x})
     $$
     
  2. 找到 Language Specific Model Masks 。用一个 0, 1 组成的 mask vector $\mathrm{M}_{s_i \rightarrow t_i} \in \{0,1\}^{|\theta|}$ 来标识一个 sub-network 。1 表示该权重有关，0表示无关。得到这个 mask vector 后，整个网络中与语言对 $s_i \rightarrow t_i$ 有关的参数即为 $\theta_{s_i \rightarrow t_i} = \{\theta_{0}^{j} | \mathrm{M}_{s_i \rightarrow t_i}^j=1 \}$ 。而得到这个 mask vector 的过程如下：
  
     <img src="https://img.wzf2000.top/image/2021/08/20/image-20210820114812132.png" alt="image-20210820114812132" style="zoom:80%;" />
  
     $\mathrm{M}_{s_i \rightarrow t_i}$ 即是把参数中权重最低的 $\alpha\%$ 置0，其余置1。
  
  3. 进行 Structure-aware Joint Training 。batches 由一个语言对的 sentence pairs 组成，在这一 batch 上计算 loss，bp 过程中只对 $\mathrm{M}_{s_i \rightarrow t_i}$​ 标识出的 sub-network 进行更新。
  
  由以上步骤，仍得到了对所有 language directions 适用的模型 $\theta^*$，在 预测中需要将 $\theta^*$ 与 $\mathrm{M}_{s_i \rightarrow t_i}$​ 联合使用。
  
- 数据集：IWSLT 与 WMT

- mask similarity 和 language family similarity 显示出较大的相关性

- 结果结论：LaSS 可以减少 parameter interference 并提升表现，对 new language pairs 的翻译有较好泛化能力；此外，LaSS 在 zero-shot translation 上比 baseline 的 BLEU 提高了惊人的 26.5 。

Question ：zero-shot 的 mask vector 是怎么做的？

- 关于 zero-shot 如何构建 mask vector ：找另外的语言作为 "bridge" ，例如若 X->Z 不存在， X->Y，Y->Z 存在，则将后两者作一个merge来获得 X->Z 的 mask vector。



## 7. Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data

- Link : [Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data - ACL Anthology](https://aclanthology.org/2021.acl-long.66/)

- Authors : Wei-Jen Ko1∗ , Ahmed El-Kishky2∗ , Adithya Renduchintala3 , Vishrav Chaudhary3 , Naman Goyal3 , Francisco Guzman´ 3 , Pascale Fung4 , Philipp Koehn5 , Mona Diab3 1University of Texas at Austin, 2Twitter Cortex, 3Facebook AI, 4The Hong Kong University of Science and Technology, 5 Johns Hopkins University

- 摘要：对 low-resource languages 构建 NMT 系统时，parallel data 的稀少是一大难题，而一些在语言学意义上与其相关、相似的 high-resource 语言则可以起到帮助。本文旨在用 NMT-Adapt 方法，结合 denoising autoencoding, back-translation, 和 adversarial objectives 来利用好 monolingual data，增加对 low-resource languages 的翻译能力。

- NMT-Adapt ：两个方向 low-resource <-> English

  - low resource -> English：典型的 unsupervised domain adaption task
  - English -> low resource ：(1) denoising autoencoder (2) adversarial training (3) high-resource translation (4) low-resource backtranslation

- NMT-Adapt 简图

  ![image-20210826012826411](https://img.wzf2000.top/image/2021/08/26/image-20210826012826411.png)

- English to low-resource：以一个 pretrained 的 mBART 为开始

  - Task 1 : Translation , 把英语翻译成 **H**igh-**R**esource **L**anguage (HRL)
  - Task 2 : Denoising Autoencoding ，加入噪声 mask 。目的是获取一个鲁棒性较好、含有语义信息的 "feature space" 。这一步对 LRL 和 HRL 都进行。
  - Task 3 : Backtranslation , 反向翻译, "capture a language-modeling effect in the low-resource language"
  - Task 4 : Adversarial Training , 对抗训练，使得 "encoder output language-agnostic features" , 使得 encoder 集中于对语义的关注而不是 language-specific information 

- Low-resource to English

  - Task 1 : Translation
  - Task 2 : Backtranslation
  - Task 3 : Adversarial Training

- Iterative Training 提高质量：

  <img src="https://img.wzf2000.top/image/2021/08/26/image-20210826015139298.png" alt="image-20210826015139298" style="zoom:150%;" />

- 结果：

  <img src="https://img.wzf2000.top/image/2021/08/26/image-20210826015339892.png" alt="image-20210826015339892" style="zoom:150%;" />



## 8. Bilingual Lexicon Induction via Unsupervised Bitext Construction and Word Alignment

- Link : [Bilingual Lexicon Induction via Unsupervised Bitext Construction and Word Alignment - ACL Anthology](https://aclanthology.org/2021.acl-long.67/)

- Authors : Haoyue Shi (TTI-Chicago) , Luke  Zettlemoyer (University of Washington, Facebook AI Research) , Sida I. Wang (Facebook AI Research) 
- 摘要：为了提高 Bilingual lexicons 的质量，本文采取混合  (1) unsupervised bitext mining and (2) unsupervised word alignment 的方式。用一个使用近期算法的 pipeline 可以显著提高 induced lexicon quality 并且学习 filter the resulting lexical entries 可以取得进一步收获。**BLI** : Bilingual Lexicon Induction

- BLI Framework 概览：

  ![image-20210826123050885](https://img.wzf2000.top/image/2021/08/26/image-20210826123050885.png)

  采用统计方法和 MLP 相结合

- Result：

  ![image-20210826124452134](https://img.wzf2000.top/image/2021/08/26/image-20210826124452134.png)



## 9. Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation

- Link : [Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation](https://aclanthology.org/2021.acl-long.91.pdf)

- 在 NMT 模型中，为了更好评估 relative source and target contributions，本文引入了 **L**ayerwise **R**elevance **P**ropagation (**LRP**) 。LRP 的特别之处在于其评估 "proportions of each token's influence" 。把 LRP 引进到 Transformer 中，对于不同的 prefixes 来分析变化，在模型训练过程中分析。

- LRP 基于的 idea ：Conservation Principle

  ![image-20210826180454791](https://img.wzf2000.top/image/2021/08/26/image-20210826180454791.png)

  （the total contribution of neurons at each layer is constant）

  ......

- 主要的成果：

  - 使用了 LRP 来评估 NMT 预测过程中 source & target 的相对贡献
  - 在不同 prefixes 上（reference, generated by a model or random translations）实验，分析了 source & target 的贡献如何变化
  - models suffering from exposure bias are more prone to over-relying on target history
  -  (i) with more data, models rely on source information more and have more sharp token contributions； (ii) the training process is non-monotonic with several distinct stages.



## 10. Improving Zero-Shot Translation by Disentangling Positional Information

- Link : [Improving Zero-Shot Translation by Disentangling Positional Information](https://aclanthology.org/2021.acl-long.101.pdf)

- Authors : Danni Liu , Jan Niehues , James Cross , Francisco Guzman , Xian Li , Department of Data Science and Knowledge Engineering, Maastricht University , Facebook AI

- 摘要：该文章发现，input tokens 中的 positional correspondence 会损害 zero-shot translation 的质量，在 middle encoder 层去除一层 residual connections 能够可观地提高 zero-translation 翻译质量。并构思了一种 Disentangling Positional Information 的模型，它可以容易地整合新语言，支持 zero-shot translation。并且通过分析，得出该模型能创造更多 language-independent representations (token & sentence level)

-  图解：
  
  ![image-20210826235458352](https://img.wzf2000.top/image/2021/08/26/image-20210826235458352.png)

  （ residual connection 可能会加强 input -> encoder's output 的对应关系： <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917010309263.png" alt="image-20210917010309263" style="zoom:33%;" /> )
  
-  采用的两种方法：
  
  - 2.1 仅在一个 encoder layer 中去除 residual connections (如，在5+5的transformer中，将第3层的残差层)
  - 2.2 把 Q 用正弦簇（如：波长100）来代替，以降低 Q 和 K 之间的相似性，减小在 self-attention 中得到的对应性（但这个方法得到的结果貌似也没前一种好，如下图）
  
- 部分结果：

  ![image-20210827002820352](https://img.wzf2000.top/image/2021/08/27/image-20210827002820352.png)
  
- 训练了两个分类器，来证实在 residual 的模型下确认位置信息更加困难：

  <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917011029477.png" alt="image-20210917011029477" style="zoom:67%;" />



## 11. Attention Calibration for Transformer in Neural Machine Translation

- Link : [Attention Calibration for Transformer in Neural Machine Translation](https://aclanthology.org/2021.acl-long.103.pdf)
- Authors : Yu Lu1,2∗ , Jiali Zeng3 , Jiajun Zhang1,2† , Shuangzhi Wu3 and Mu Li3 1 National Laboratory of Pattern Recognition, Institute of Automation, CAS, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3 Tencent Cloud Xiaowei, Beijing, China
- 摘要：在基于 Transformer 的 NMT 中，attention 机制对发现 decisive inputs 的能力需要质疑，本文提出了 一种校准的方法。引入一个 **mask perturbation model** ，它可以自动评估每个 input 对 output 的 contribution 。对于那些 indispensable tokens ，增加其 attention 权重 （focus on key inputs）。

- calibration：

  ![image-20210827113221428](https://img.wzf2000.top/image/2021/08/27/image-20210827113221428.png)

  上图，mask perturbation model 认为“远郊” 是 decisive input ，故加强其权重

  下图，在校准前，翻译完 death 之后 EOS 的权重较重，这可能会造成“交通中断”的漏译；理应加强 attention 权重较低确更 informative 的 “交通中断” 

- mask perturbation model ：

  ![image-20210827114753592](https://img.wzf2000.top/image/2021/08/27/image-20210827114753592.png)

- 通过 mask 的方法搜索得到最 informative 的 inputs 。通过三种融合方法来将校准后的 attention 注入原先的  attention weights :

  - fixed weighted sum
  - annealing learning
  - gating machanism

  Mask Perturbation Model 与 NMT model（Attention Calibration Network）是 jointly trained

  <img src="https://img.wzf2000.top/image/2021/08/27/image-20210827115123026.png" alt="image-20210827115123026" style="zoom:150%;" />

- 实验中还发现，校准的 attention 权重在较低层更均匀，在较高层更 focused 。"High entropy attention weights are found to have great needs for calibration at all layers " 。

- results :

  ![image-20210827115503196](https://img.wzf2000.top/image/2021/08/27/image-20210827115503196.png)



## 12. Diverse Pretrained Context Encodings Improve Document Translation

- Link : [Diverse Pretrained Context Encodings Improve Document Translation](https://aclanthology.org/2021.acl-long.104.pdf)

- Authors : Domenic Donato, Lei Yu, Chris Dyer (DeepMind)
- 摘要：本文设计了一个架构， 整合来自多个预训练文本的 context signals，来适应一个 sentence-level 的 seq2seq transformer ，并评估了 (1) 生成这些信号的不同预训练方法对翻译性能的影响 (2) 并行数据的量 (3) conditioning on source, target or source&target context 。
- 主要结论：(1) 多种源语言的 context 可以提高文档级别翻译的性能；(2) 平行上下文数据的数量对于性能至关重要；(3)  source language context 更有价值（除非target的质量非常高）。

- Multi-context Model :

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021060821463241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNDg1Mjcz,size_16,color_FFFFFF,t_70)



## 13. Contributions of Transformer Attention Heads in Multi- and Cross-lingual Tasks

- Link : [Contributions of Transformer Attention Heads in Multi- and Cross-lingual Tasks](https://aclanthology.org/2021.acl-long.152.pdf)

- 摘要：分析了 Transformer 模型中不同 attention heads 的相对重要性，以及在 cross-lingual / multi-lingual 任务中的可解释性。通过实验，可以得出：在 multi-lingual Transformer-based model 中对一部分 attention heads 剪枝，基本上都有正面效应；可以用 rank using gradients 和 identify with a few trial experiments 的方法来确定被剪枝的 attention heads 。

- Head rankings :

  1. We fine-tune a Transformer-based model on a mono-lingual task for three epochs

  2. We re-run the fine-tuned model on the development partition of the dataset with back-propagation but not parameter updates to obtain gradients

  3. We sum up the absolute gradients on each head, layer-wise normalize the accumulated gradients, and scale them into the range [0, 1] globally

     ![image-20210901003710793](https://img.wzf2000.top/image/2021/09/01/image-20210901003710793.png)

- ranking 后，把 lowest-ranked head 给剪枝，不断增加剪枝数量直至一个 limit 或直至性能开始下降

- 剪枝结果

  ![image-20210901003418135](https://img.wzf2000.top/image/2021/09/01/image-20210901003418135.png)



## 14. Crafting Adversarial Examples for Neural Machine Translation

- Link : [Crafting Adversarial Examples for Neural Machine Translation](https://aclanthology.org/2021.acl-long.153.pdf)

- Authors : Xinze Zhang， Junzhe Zhang， Zhenhua Chen， Kun He （Huazhong University of Science and Technology）

- 摘要：尝试对 NMT adversarial attacks 作了评估，给出了 NMT Adversarial Example 的数学定义，并且设计了一种精巧的制造 NMT Adversarial Example 的方法（ A black-box attack method : **W**ord **S**aliency speedup **L**ocal **S**earch , **WSLS** ），该方法生成的例子展现了显著的 attack 主流 NMT 模型的能力。

- adversarial example

  ![image-20210901005434115](https://img.wzf2000.top/image/2021/09/01/image-20210901005434115.png)

  为了增强 NMT system 的鲁棒性，制造这些 adversarial examples 是值得考虑的。

  好的 adversarial examples 应该在语法上保持准确，句子流畅，不出拼写错误，但用微小的 perturbation 来 "destroy the translation performance" 。

### (1) : Definition of NMT adversarial example

- Definition : NMT adversarial example

  ![image-20210903151247181](https://img.wzf2000.top/image/2021/09/03/image-20210903151247181.png)

  ( St(·, ·) is a metric for evaluating the similarity of two sentences )

  即新句子 x' 与原句子 x 在 embedding 空间上距离足够近，原翻译和参考翻译足够相似，x 的 round-trip 翻译与原句子足够相似，但是 x' 与其 round-trip 翻译的相似性与原句子x 相比有较大出入。

  ![image-20210901101130797](https://img.wzf2000.top/image/2021/09/01/image-20210901101130797.png)

- 用 Mean Decrease (MD) 和 Mean Percentage Decrease (MPD) 来评估 adversary ：

  ![image-20210901010526571](https://img.wzf2000.top/image/2021/09/01/image-20210901010526571.png)

  

### (2) : WSLS Attack (Word Saliency speedup Local Search)

![image-20210901010935157](https://img.wzf2000.top/image/2021/09/01/image-20210901010935157.png)

- Part A : Candidates Generator (phase1)

  对每个词，使用 KNN 来找 embedding 空间中 k 个最近的词作为 candidates  。

- Part B : GOGR (phase1)

  采用 GOGR (Greedy Order Greedy Replacement) 的策略来选择替代词。在每一步都枚举所有可能的、未 attack 的位置，对每个位置都用贪心策略![image-20210901011720709](https://img.wzf2000.top/image/2021/09/01/image-20210901011720709.png)找到替换词，再选择最好的整句 w* ：![image-20210901011837962](https://img.wzf2000.top/image/2021/09/01/image-20210901011837962.png)，迭代替换直至替换了足够的词语。

- Part C : Saliency Generator

  使用 MLM (Mask Language Model) 来帮助计算 Saliency scores 。word saliency 显示一个词被 mask 后整个句子句意变化的程度。

  ![image-20210901012020442](https://img.wzf2000.top/image/2021/09/01/image-20210901012020442.png)

  从中看出，更高的 word saliency 表示该词与上下文关系越小，这样的词更好 attack 。

- Part D : Local Search Strategy (phase2)

  前面的部分获取了一个 initial example : x' 。而这一部分则是要进行局部搜索，迭代地对这个 adversarial example 进行优化。

  这部分分为三种 walk : Saliency Walk , Random Walk , Certain Walk 。

  - Saliency Walk ：根据 saliency 升序，mute 掉 perturbed words (还原)，higher saliency 的 words 有 higher priority ；根据 saliency 降序，prune 掉 unperturbed words (剪枝)，迭代找最优 。有一个 early stop 策略。
  - Random Walk ：为了防止当前的 adversarial example 陷入局部最优，随机 mute 掉一个 perturbed word ，找到最优的。
  - Certain Walk ：在 random 之后充分探索，去除剪枝操作，增大搜索领域。

  采取 {SW, RW, CW, RW, CW} 轮回的操作，并引入 early stop 策略。

  ![image-20210903151324492](https://img.wzf2000.top/image/2021/09/03/image-20210903151324492.png)

Experiments on : RNN & Transformer

- MD & MPD :

![image-20210901013916248](https://img.wzf2000.top/image/2021/09/01/image-20210901013916248.png)

- BLEU degradation :

![image-20210901014026180](https://img.wzf2000.top/image/2021/09/01/image-20210901014026180.png)

- Examples

  ![image-20210901101146206](https://img.wzf2000.top/image/2021/09/01/image-20210901101146206.png)

**不同 attack 标准** ， **不好评估**



## 15. UXLA: A Robust Unsupervised Data Augmentation Framework for Zero-Resource Cross-Lingual NLP

- Link : [UXLA: A Robust Unsupervised Data Augmentation Framework for Zero-Resource Cross-Lingual NLP](https://aclanthology.org/2021.acl-long.154.pdf)

- Authors : M Saiful Bari ∗ ¶ , Tasnim Mohiuddin ∗¶, and Shafiq Joty¶§ ¶ Nanyang Technological University, Singapore § Salesforce Research Asia, Singapore

- 摘要：为了解决 low-resource / zero-resource 缺少数据的问题，提出了一个 **UXLA** (unsupervised cross-lingual augmentation) 模型，来进行数据增强以及无监督的样例选取。该模型在 XNER、XNLI、PAWS-X 数据集上都有不错的提高，并分析了其性能提升的因素。

- UXLA : ![image-20210903020806374](https://img.wzf2000.top/image/2021/09/03/image-20210903020806374.png)

- 主要步骤：

  - Warm-up: Training Task Models
  - Sentence Augmentation
  - Co-labeling through Co-distillation
  - Data Samples Manipulation

- 成果：

  ![image-20210903021700132](https://img.wzf2000.top/image/2021/09/03/image-20210903021700132.png)

<img src="https://img.wzf2000.top/image/2021/09/03/image-20210903021724260.png" alt="image-20210903021724260" style="zoom:80%;" />

<img src="https://img.wzf2000.top/image/2021/09/03/image-20210903021739528.png" alt="image-20210903021739528" style="zoom:80%;" />



## 16. Glancing Transformer for Non-Autoregressive Neural Machine Translation

- Link ：[Glancing Transformer for Non-Autoregressive Neural Machine Translation](https://aclanthology.org/2021.acl-long.155.pdf)

- Authors ：Lihua Qian1,2∗ Hao Zhou2 Yu Bao3 Mingxuan Wang2 Lin Qiu1 Weinan Zhang1 Yong Yu1 Lei Li2 1 Shanghai Jiao Tong University 2 ByteDance AI Lab 3 Nanjing University

- 摘要：以往的 non-autoregressive transformer (NAT) 在表现上和 Transformer 仍有一定差距，本文提出一种表现近似 Transformer 而又是 non-autoregressive ，在 infer 时只需要 one pass of decoding (高效) 的模型—— **G**lancing **L**anguage **M**odel , 并在此基础上建立了 **G**lancing **T**ransformer(**GLAT**) 。

- 这个模型采取了一种 "adaptive glancing sampling strategy" 

  ![image-20210903024057909](https://img.wzf2000.top/image/2021/09/03/image-20210903024057909.png)

  ![image-20210903024108891](https://img.wzf2000.top/image/2021/09/03/image-20210903024108891.png)

- "GLAT does not modify the network architecture, which is a training method to explicitly learn word interdependency"

- 结果：

  ![image-20210903024240074](https://img.wzf2000.top/image/2021/09/03/image-20210903024240074.png)



## 17. Self-Training Sampling with Monolingual Data Uncertainty for Neural Machine Translation

- Link : [Self-Training Sampling with Monolingual Data Uncertainty for Neural Machine Translation](https://aclanthology.org/2021.acl-long.221.pdf)

- Authors : Wenxiang Jiao†∗ Xing Wang‡ Zhaopeng Tu‡ Shuming Shi‡ Michael R. Lyu† Irwin King† †Department of Computer Science and Engineering The Chinese University of Hong Kong, HKSAR, China ‡Tencent AI Lab

- 摘要：在 self-training 中，为了增加平行数据常需要进行随机采样，本文希望优化采样，通过计算 "uncertainty" 来选取 "most informative" 的 monolingual sentences (这些样本通常具有 higher uncertainty) 。结果显示对 high-uncertainty 句子和 low-frequency 词语的翻译都有提升。

- 主要贡献：

  - 展示了 self-training 中 distinguish monolingual sentences 的重要性
  - 设计了一个 uncertainty-based sampling strategy
  - 展示出 NMT 模型在 self-training 中从 uncertain monolingual sentences 获得更大收益

- uncertainty ：

  ![image-20210903031809734](https://img.wzf2000.top/image/2021/09/03/image-20210903031809734.png)

- 框架：

  ![image-20210903031918559](https://img.wzf2000.top/image/2021/09/03/image-20210903031918559.png)

- 结果：

  ![image-20210903031956228](https://img.wzf2000.top/image/2021/09/03/image-20210903031956228.png)



## 18. Breaking the Corpus Bottleneck for Context-Aware Neural Machine Translation with Cross-Task Pre-training

- Link : [Breaking the Corpus Bottleneck for Context-Aware Neural Machine Translation with Cross-Task Pre-training](https://aclanthology.org/2021.acl-long.222.pdf)

- Authors : Linqing Chen 1Junhui Li∗ 1Zhengxian Gong 2Boxing Chen 2Weihua Luo 1Min Zhang 1Guodong Zhou 1School of Computer Science and Technology, Soochow University, Suzhou, China 2Alibaba DAMO Academy

- 摘要：为了提升 context-aware NMT ，本文选择利用其大规模句子级别的平行数据以及源 source-side 单语言文档级别数据。为此，设计两个预训练任务：(1) 在句级别平行数据上进行 source->target 的翻译 (2) 在单语文档上把 deliberately noised 的文档翻译成原文档 。这两个任务在同一个模型上联合预训练，并在之后进行 fine-tuning 。实验结果显示该方法显著提升翻译表现，并可以兼顾句级别、文档级别翻译。

-  Task ：

  ![image-20210903113402631](https://img.wzf2000.top/image/2021/09/03/image-20210903113402631.png)

- Model :

  ![image-20210903113455875](https://img.wzf2000.top/image/2021/09/03/image-20210903113455875.png)

- 部分结果：

  ![image-20210903114046119](https://img.wzf2000.top/image/2021/09/03/image-20210903114046119.png)



## 19. Guiding Teacher Forcing with Seer Forcing for Neural Machine Translation

- Link : [Guiding Teacher Forcing with Seer Forcing for Neural Machine Translation](https://aclanthology.org/2021.acl-long.223.pdf)

- Authors : Yang Feng1,2 Shuhao Gu1,2 Dengji Guo1,2 Zhengxin Yang1,2 Chenze Shao1,2 ∗ 1 Key Laboratory of Intelligent Information Processing Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS) 2 University of Chinese Academy of Sciences, Beijing, China

- 摘要 : teacher forcing 通常限定于 past information 来做 predictions ，缺乏 global planning for future 。本文引入了一种新的 **seer decoder** ，在 target prediction 中引入了 future information 。同时，让 conventional decoder 通过知识蒸馏模拟 seer decoder 的行为。结果显示 seer decoder 又显著的提升，并且知识蒸馏是 transfer knowledge distillation from the seer decoder to the conventional decoder 的最好途径。

- 思路：

  ![image-20210903121334205](https://img.wzf2000.top/image/2021/09/03/image-20210903121334205.png)
  
- seer decoder :

  ![image-20210904021924829](https://img.wzf2000.top/image/2021/09/04/image-20210904021924829.png)

- 部分结果 :

  ![image-20210904021945363](https://img.wzf2000.top/image/2021/09/04/image-20210904021945363.png)



## 20. Online Learning Meets Machine Translation Evaluation: Finding the Best Systems with the Least Human Effort

- Link : [Online Learning Meets Machine Translation Evaluation: Finding the Best Systems with the Least Human Effort](https://aclanthology.org/2021.acl-long.242.pdf)

- Vania Mendonc¸a ˆ 1,2 , Ricardo Rei1,2,3 , Lu´ısa Coheur1,2 , Alberto Sardinha1,2 , Ana Lucia Santos ´ 4,5 1 INESC-ID Lisboa, Portugal 2 Instituto Superior Tecnico, Universidade de Lisboa, Portugal ´ 3 Unbabel AI, Lisboa, Portugal 4 Centro de Lingu´ıstica da Universidade de Lisboa, Portugal 5 Faculdade de Letras da Universidade de Lisboa, Portugal

- 摘要：本文提出了通过 online learning 收敛得到 best MT system 的一个框架（以往的自动评价指标如 BLEU ，其效果不如人工评估，而人工评估的代价又太大）。该框架引入了 expert device (weights) 以及 multi-armed bandits (多臂老虎机) 来实现 online learning 。

- Online Learning Process

  <img src="https://img.wzf2000.top/image/2021/09/10/image-20210910013000634.png" alt="image-20210910013000634" style="zoom:67%;" />

- 部分结果 (WMT19) ：

  <img src="https://img.wzf2000.top/image/2021/09/10/image-20210910013238132.png" alt="image-20210910013238132" style="zoom:50%;" />

  

- Comments : 把一些机器学习方法用于评估 MT 模型，思路感觉以前没见到过，但做法还是比较传统的。并且用的数据集好像也不是很大...？



## 21. From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text

- Link : [From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.acl-long.245.pdf)

- Author : Ishan Tarunesh , Syamantak Kumar , Preethi Jyothi Samsung Korea, Google India, IIT Bombay

- 摘要：生成 **C**ode-**S**witched text （在同一句话中实验不止一种语言的现象） 是一个重要的任务。而 CS 的语料数据是缺乏的，故本文做了一个生成高质量 code-switched text 的工作，使用 NMT 模型从 Hindi 单语言语料生成 Hindi-English switched code 。本文展示了将生成的文本用于下游 code-switching NLP task 的 improvement 。

- Example：

  <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917100537731.png" alt="image-20210917100537731" style="zoom:67%;" />

- Model ：

  <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917100621551.png" alt="image-20210917100621551" style="zoom:67%;" />



## 22. Fast and Accurate Neural Machine Translation with Translation Memory

- Link：[Fast and Accurate Neural Machine Translation with Translation Memory](https://aclanthology.org/2021.acl-long.246.pdf)
- Author ：Qiuxiang He , Guoping Huang , Qu Cui , Li Li , Lemao Liu : Southwest University , Tencent AI Lab , Nanjing University

- 摘要：以往的 **T**ranslation **M**emory-based NMT 基本只在 TM-specialized 任务上有优势，并且计算开销大。本文构建了一个基于 Transformer 的 TM-based NMT ，结构简单、快速准确；针对 TM-based 模型在处理与 TM 相似性小的输入时表现下降的robustness 问题，本文采用一个好的 training criterion 来有效优化参数。该模型在 6 个 TM-specialized 任务上超过了 TM 的 baseline ，并在  Zh→En 、En→De 这 2 个 general task 上都超越了 baseline 。

- Model：

  ![image-20210917112529050](https://img.wzf2000.top/image/2021/09/17/image-20210917112529050.png)

  Three methods :

  - sentence (TF-S)
  - sentence with score (TF-SS)
  - sentence with alignment (TF-SA)

- 部分结果：

  <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917145854239.png" alt="image-20210917145854239" style="zoom:50%;" />

  <img src="https://img.wzf2000.top/image/2021/09/17/image-20210917145910168.png" alt="image-20210917145910168" style="zoom:50%;" />



## 23. G-Transformer for Document-Level Machine Translation

- Link : [G-Transformer for Document-Level Machine Translation](https://aclanthology.org/2021.acl-long.267.pdf)

- Author : Guangsheng Bao , Yue Zhang , Zhiyang Teng , Boxing Chen , Weihua Luo - School of Engineering , Westlake University ， Institute of Advanced Technology , Westlake Institute for Advanced Study ， DAMO Academy , Alibaba Group Inc.

- 摘要：本文发现，文档级别翻译的失败不是来自过拟合，而是在训练中陷入局部最小。复杂度不断增加的 target-to-source attention 是其一个原因。为了解决这个问题，本文提出了 **G-Transformer** ，引入局部性假设作为一个 inductive bias 。G-Transformer 可以更快、更稳定地收敛，在多个 benchmarks 上的 BLEU 指标上达到了 state of the art 。可以说该工作是第一个 document-by-document 的 translation model 。

- Structure : 

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922134447216.png" alt="image-20210922134447216" style="zoom: 50%;" />

  - a 方法运行复杂度高，并且缺少上下文信息
  - b 方法若把整个文档进行 encoding 会 fail，多个句子时需引入 data augmentation，训练时间长
  - c 方法，使用 guidance mechanism ，将 self-attention 和 target-to-source attention 限制在了局部上下文。整个 document 仍是一个输入单元，group tags ①②③ 用来标注不同句子的位置；target-to-source attention 由将 target 的 tag 对应到 source 的 tag 来指导，这样一来，attention 的假设空间就缩小了。
  
- G-Transformer example ：

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922165450924.png" alt="image-20210922165450924" style="zoom:80%;" />

- Group Tags :

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922165508574.png" alt="image-20210922165508574" style="zoom: 50%;" />

  

  用 $G_x$ 和 $G_y$ 来表示 X 与 Y 之间的 alignment 。

- Group Attention :

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922202734341.png" alt="image-20210922202734341" style="zoom:60%;" />

  $I_Q$ 为长度和 $G_Q$ 相同的全 1 向量，$I_K$ 类似。

  而 Group Multi-Head Attention 为 ：

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922204256992.png" alt="image-20210922204256992" style="zoom:60%;" />

- Combined Attention :

  <img src="https://img.wzf2000.top/image/2021/09/22/image-20210922204940073.png" alt="image-20210922204940073" style="zoom:80%;" />

- 部分结果：

  ![image-20210922205035036](https://img.wzf2000.top/image/2021/09/22/image-20210922205035036.png)



## 24. Prevent the Language Model from being Overconfident in Neural Machine Translation

- Link : [Prevent the Language Model from being Overconfident in Neural Machine Translation](https://aclanthology.org/2021.acl-long.268.pdf)
- Authors : Mengqi Miao , Fandong Meng , Yijin Liu , Xiao-Hua Zhou , Jie Zhou - Peking University, China; Pattern Recognition Center, WeChat AI, Tencent Inc, China; Beijing International Center for Mathematical Research, National Engineering Lab for Big Data Analysis and Applications, Department of Biostatistics, Peking University, Beijing, China

- 摘要：NMT 有时会出现 inadequate translation 的问题，分析认为可能是 language model 的 overconfidence 导致。本文定义了 **Margin between the NMT and the LM** ，该指标与 LM 的 overconfidence degree 反相关。为了最大化 margin ，本文提出了两种方法，Margin-based Token-level Objective(**MTO**) 与 Margin-based Sentence-level Objective(**MSO**) ，防止 LM 的 overconfidence 。在 WMT 的部分数据集上提高了零点几至一点几的 BLEU 。

- Margin between the NMT and the LM：

  ![image-20210924105203093](https://img.wzf2000.top/image/2021/09/24/image-20210924105203093.png)

- 部分结果：

  ![image-20210924130356626](https://img.wzf2000.top/image/2021/09/24/image-20210924130356626.png)

- Comments ：主要只做了en-de , zh-en 和 en-fr 这三个语言对，并且 BLEU 提升不高，感觉定义的一些 loss 等比较 tricky ，不过 margin 是个有趣的视角。



## 25. Point, Disambiguate and Copy: Incorporating Bilingual Dictionaries for Neural Machine Translation

- Link : [Point, Disambiguate and Copy: Incorporating Bilingual Dictionaries for Neural Machine Translation](https://aclanthology.org/2021.acl-long.307.pdf)

- Author : Tong Zhang1,2 , Long Zhang1,2 , Wei Ye1,† , Bo Li1,2 , Jinan Sun1 , Xiaoyu Zhu3 , Wen Zhao1 , Shikun Zhang1,† 1 National Engineering Research Center for Software Engineering, Peking University 2 School of Software and Microelectronics, Peking University 3 BIGO

- 本文提出了一个将双语词典整合进 NMT 的复杂框架：PDC。采用了三个组件：**P**ointer，**D**isambiguator 和 **C**opier 。这个框架在中英、中日 benchmarks 上显示了其效果。

- Example :

  ![image-20210924112434373](https://img.wzf2000.top/image/2021/09/24/image-20210924112434373.png)

  - Pointer : 利用双语词典中的语义信息
  - Disambiguator : 从 source 和 target 合成上下文信息
  - Copier : 基于分层复制机制系统地连接以上两者

- Structure : 

  ![image-20210924125058793](https://img.wzf2000.top/image/2021/09/24/image-20210924125058793.png)

- 部分结果：

  ![image-20210924130555492](https://img.wzf2000.top/image/2021/09/24/image-20210924130555492.png)



## 26. Towards User-Driven Neural Machine Translation

- Link : [Towards User-Driven Neural Machine Translation](https://aclanthology.org/2021.acl-long.310.pdf)

- Authors : Huan Lin1,2 Liang Yao3 Baosong Yang3 Dayiheng Liu3 Haibo Zhang3 Weihua Luo3 Degen Huang4 Jinsong Su1,2,5∗ 1School of Informatics, Xiamen University 2 Institute of Artificial Intelligence, Xiamen University 3Alibaba Group 4Dalian University of Technology 5Pengcheng Lab, Shenzhen

- 摘要：本文进行了对 user-driven NMT 的探索，基于 cache module 和 contrastive learning 设计了一个 user-driven NMT ，可以融入不同用户的特点。还收集建立了一个 **U**ser-**D**riven Machine **T**ranslation 数据集：**UDT-Corpus** 。

- Model :

  ![image-20210928163907684](https://img.wzf2000.top/image/2021/09/28/image-20210928163907684.png)

- Data Annotation : 从在线翻译系统收集数据后，构建一个三元组：$<X^{(u)}, Y^{(u)}, H^{(u)}>$​ ，X 表示 source sentence ，H 表示用户 u 的历史输入，Y 表示 target translation sentence 。

- User-Driven NMT : 

  - Cache-based User Behavior Modeling : 

    - topic cache ：该用户的长期特点
    - context cache ：该用户的短期特点，近期的输入

  - Cache representation ：TF-IDF，以该用户的历史输入作为 document ，以所有用户的所有输入作为 corpus

  - 将 user 特点加入 embedding ：

    <img src="https://img.wzf2000.top/image/2021/09/28/image-20210928175901511.png" alt="image-20210928175901511" style="zoom: 33%;" />

  - Training with a Contrastive Loss ：

    <img src="https://img.wzf2000.top/image/2021/09/28/image-20210928180138718.png" alt="image-20210928180138718" style="zoom: 33%;" />

    $L_{cl}$ ：

    <img src="https://img.wzf2000.top/image/2021/09/28/image-20210928180240136.png" alt="image-20210928180240136" style="zoom: 33%;" />

- 主要结果：

  <img src="https://img.wzf2000.top/image/2021/09/28/image-20210928180544312.png" alt="image-20210928180544312" style="zoom:33%;" />

- Example : 

  <img src="https://img.wzf2000.top/image/2021/09/28/image-20210928180832573.png" alt="image-20210928180832573" style="zoom: 40%;" />



## 27. End-to-End Lexically Constrained Machine Translation for Morphologically Rich Languages

- Link : [End-to-End Lexically Constrained Machine Translation for Morphologically Rich Languages](https://aclanthology.org/2021.acl-long.311.pdf)

- Authors : Josef Jon and João Paulo Aires and Dušan Variš and Ondrej Bojar ˇ Charles University

- 摘要 : 对于形态丰富的语言，词汇受限翻译容易造成上下文不通的问题。本文尝试使用把限制词与 source sentence 放到一起的办法来进行处理。在英语-捷克语的词汇受限翻译任务上减少了 inflection error ，提升了翻译质量。

- 方法：

  ![image-20210930013355485](https://img.wzf2000.top/image/2021/09/30/image-20210930013355485.png)

  <sep> 来分割 input 与 constraints ，<c> 分割 constraints 

  并使用 bilingual dictionary 。

- 部分结果：

  ![image-20210930013432146](https://img.wzf2000.top/image/2021/09/30/image-20210930013432146.png)



## 28. SemFace: Pre-training Encoder and Decoder with a Semantic Interface for Neural Machine Translation

- Link : [SemFace: Pre-training Encoder and Decoder with a Semantic Interface for Neural Machine Translation](https://aclanthology.org/2021.acl-long.348.pdf)

- Authors : Shuo Ren†‡∗ , Long Zhou‡ , Shujie Liu‡ , Furu Wei‡ , Ming Zhou‡ , Shuai Ma† †SKLSDE Lab, Beihang University, Beijing, China ‡Microsoft Research Asia, Beijing, China

- 摘要：encoder-decoder 结构中，enc/dec 间的 cross-attention 没办法预训练。而本文设计了一个 semantic interface (**SemFace**) 在预训练的 encoder 和预训练的  decoder 之间。设计了两种 SemFace : CL-SemFace (cross-lingual) 与 VQ-SemFace (vector quantized) 。在许多对语言的数据上达到了 bleu 的提升，尤其是 low resource 以及 unsupervised 语言对。

- SemFace :

  ![image-20210930113324440](https://img.wzf2000.top/image/2021/09/30/image-20210930113324440.png)

- CL-SemFace :

  ![image-20210930113749973](https://img.wzf2000.top/image/2021/09/30/image-20210930113749973.png)

- VQ-SemFace :

  ![image-20210930113851328](https://img.wzf2000.top/image/2021/09/30/image-20210930113851328.png)



## 29. Energy-Based Reranking: Improving Neural Machine Translation Using Energy-Based Models

- Link : [Energy-Based Reranking: Improving Neural Machine Translation Using Energy-Based Models](https://aclanthology.org/2021.acl-long.349.pdf)
- Affiliation : University of North Carolina Charlotte,  University of Massachusetts Amherst
- Author : Sumanta Bhattacharyya, Amirmohammad Rooshenas, Subhajit Naskar, Simeng Sun, Mohit Iyyer, and Andrew McCallum
- 摘要：MLE 用于 autoregressive NMT 的训练，而 measure 常采用 bleu 等指标，本文注意到了这一不匹配的情况。而本文作者就此探索了一种参数逼近的度量。采用 energy-based models (EBMs) 来进行参数化，训练了一个 energy-based model 来模拟 measure ，总结了一个基于 EBR 的 re-ranking 算法：energy-based re-ranking (**EBR**) 。在 transformer-based nmt 上取得了好的训练效果。
- ![image-20211006205002518](https://img.wzf2000.top/image/2021/10/06/image-20211006205002518.png)
- <img src="https://img.wzf2000.top/image/2021/10/06/image-20211006205023938.png" alt="image-20211006205023938" style="zoom:50%;" />
- Comments : 引入了 EBM 的方法，比较有新意



## 30. On Compositional Generalization of Neural Machine Translation

- Link : [On Compositional Generalization of Neural Machine Translation](https://aclanthology.org/2021.acl-long.368.pdf)

- Affiliation : Zhejiang University, Westlake University, Westlake Institute for Advanced Study

- Authors : Yafu Li~ , Yongjing Yin~ , Yulong Chen~ , Yue Zhang

- 摘要：本文构建了一个216k个干净、流畅的句子对的数据集 **CoGnition** ，并在该数据集上研究了 NMT 模型的 compositional generalization 。

- compositional generalization : The ability to produce a potentially infinite number of novel combinations of known components

- CoGnition （En-Zh) : 

  ![image-20211007010445901](https://img.wzf2000.top/image/2021/10/07/image-20211007010445901.png)

  出了 train/valid/test 数据集，该数据集还包含了一个 compositional generalization test set ，每句都包含 novel compounds

  ~~（有点prompt的感觉？）~~

- 主流的 transformer NMT 模型在一些novel成分的翻译上面临挑战，这在BLEU上是体现不出来的

- dataset 包含有不同成分 novel compounds 的句子

  ![image-20211007011759655](https://img.wzf2000.top/image/2021/10/07/image-20211007011759655.png)

- Experiments : We conduct experiments on CoGnition dataset and perform human evaluation on the model results.

- 部分结果：

  <img src="https://img.wzf2000.top/image/2021/10/07/image-20211007012025095.png" alt="image-20211007012025095" style="zoom:50%;" />



## 31. Mask-Align: Self-Supervised Neural Word Alignment

- Link : [Mask-Align: Self-Supervised Neural Word Alignment](https://aclanthology.org/2021.acl-long.369.pdf)

- Affiliation : Tsinghua University, Beijing National Research Center for Information Science and Technology, Beijing Academy of Artificial Intelligence

- Authors : Chi Chen , Maosong Sun , Yang Liu

- 摘要：词语对齐是寻找源语言与目标语言中对应词汇的任务，在许多自然语言处理任务中起到重要作用。目前无监督神经词语对齐侧重于从神经机器翻译模型中推断对齐结果，这种做法无法充分利用目标端的完整上下文。为此，我们提出了一个利用目标端完整上下文的自监督词语对齐模型Mask-Align。本文的模型并行地遮盖每一个目标端词，并根据源端和其余目标端词来预测它。在这一过程中，我们假设对恢复被遮盖词最有帮助的源端词应该被对齐。我们还引入了一种注意力的变体——泄漏性注意力（leaky attention）来缓解在一些特定词如句号上的过大的交叉注意力权重。在四种语言对的实验上，我们的方法都取得了最佳结果，并显著超越其他无监督神经词语对齐方法。

- Example : 

  <img src="https://img.wzf2000.top/image/2021/10/08/image-20211008110112398.png" alt="image-20211008110112398" style="zoom:50%;" />

- Architecture : 

  ![image-20211008110215165](https://img.wzf2000.top/image/2021/10/08/image-20211008110215165.png)

- Main Results : 

  <img src="https://img.wzf2000.top/image/2021/10/08/image-20211008110815668.png" alt="image-20211008110815668" style="zoom:80%;" />



## 32. GWLAN: General Word-Level Autocompletion for Computer-Aided Translation

- Link : [GWLAN: General Word-Level AutocompletioN for Computer-Aided Translation](https://aclanthology.org/2021.acl-long.370.pdf)

- Affiliation : Tencent AI Lab

- Author : Huayang Li, Lemao Liu, Guoping Huang, Shuming Shi

- 摘要：计算机辅助翻译（CAT），即在翻译过程中使用软件来协助人工翻译，已被证明有助于提高人工译员的生产力。其中根据人工译员提供的文本片段提示翻译结果的自动补全功能是CAT的核心。此前在这方面的研究有两个限制。首先，关于这个方向的大多数研究工作都集中在句子级的自动补全（即根据人工译员的输入生成整个译文），但到目前为止，词级自动补全还没有被充分探索。其次，几乎没有公开的 benchmark 可用于CAT的自动补全任务。这可能是CAT的研究进展比自动翻译慢的原因之一。在本文中，我们从真实的CAT场景中提出了一个通用词级自动补全任务（GWLAN），并构建了第一个公开基准以促进该领域的研究。此外，我们为GWLAN任务提出了一种简单有效的方法，并将其与几个 baseline 进行比较。实验证明，在构建的基准数据集上，我们提出的方法可以比 baseline方法提供更准确的预测。

- CAT example :

  <img src="https://img.wzf2000.top/image/2021/10/14/image-20211014010210339.png" alt="image-20211014010210339" style="zoom:50%;" />

- 4 种 context ：prefix , suffix , zero context & bidirectional context (propose a joint training strategy to optimize the model parameters on different types of contexts together.)

  <img src="https://img.wzf2000.top/image/2021/10/14/image-20211014011536167.png" alt="image-20211014011536167" style="zoom: 67%;" />

- 三元组 $(x,s,c)$ ：

  $x = (x_1, x_2, ... , x_m)$ 为 source sentence

  $s = (s_1,s_2,...,s_k)$ 为 sequence of human typed characters

  $c = (c_l, c_r)$ 为 $s$​ 的上下文（左、右） 

  GWLAN : 给定一个 $x$ , $s$ , $c$ ，预测出 target word $w$ 放在 $c_l$ 和 $c_r$​ 之间来补全翻译

- 构造 benchmark : 三元组 + $w$ ，4 types of context ，train/valid/test

- evaluation metric ：accuracy (ratio of matched words)

- approach : 

  - task

    - 1. model the distribution of $w$
    - 2. find the most possible

  - Word Prediction Model (WPM)

    <img src="https://img.wzf2000.top/image/2021/10/14/image-20211014012203324.png" alt="image-20211014012203324" style="zoom:67%;" />

    The cross-lingual encoder is similar to the Transformer decoder, while the only difference is that we replace the auto-regressive attention (ARA) layer by a bidirectional masked attention (BMA) module
    
    - Embeddings & Bidirectional Masked Attention
    
    ![image-20211014012308022](https://img.wzf2000.top/image/2021/10/14/image-20211014012308022.png)



## 33. Rewriter-Evaluator Architecture for Neural Machine Translation

- Link : [Rewriter-Evaluator Architecture for Neural Machine Translation](https://aclanthology.org/2021.acl-long.443.pdf)

- Affiliation : Tencent AI Lab / Ant Group

- Author : Yangming Li , Kaisheng Yao

- 摘要：现有NMT由于缺乏 multi-pass process 的终止策略导致性能不够。本文设计一种 Rewriter-Evaluator 架构。在每一步的时候，rewriter 在旧 translation 基础上生成一个新的、更好的 translation ，然后 evaluator 对其进行翻译质量的评估，依据打分结果来觉得是否终止。并且引入 prioritized gradient descent (PGD) 来进行联合训练。

- Architecture :

  Rewriter 和 Estimator 进行联合训练 

  ![image-20211014144214147](https://img.wzf2000.top/image/2021/10/14/image-20211014144214147.png)

- Comments : 有点像 GAN



## 34. Modeling Bilingual Conversational Characteristics for Neural Chat Translation

- Link : [Modeling Bilingual Conversational Characteristics for Neural Chat Translation](https://aclanthology.org/2021.acl-long.444.pdf)

- Affiliation : Beijing Jiaotong University , WeChat AI, Tencent Inc

- Author : Yunlong Liang , Fandong Meng , Yufeng Chen , Jinan Xu1, Jie Zhou

- 摘要：Neural chat translation 是用来进行双语对话的翻译的。本文构建了一个模型，设计三个隐模块，来学习对话中的 role preference , dialogue coherence 与 translation consistency ，并将其融合进 NMT 中（使用 conditional VAE 来。收集构建了中英对话数据集 BMLED 。实验显示该方法比 baseline 提高许多，并在 BLEU 和 TER 上比一些 sota 的 context-aware NMT 模型要好。

- Bilingual Conversation Example : 

  <img src="https://img.wzf2000.top/image/2021/10/28/image-20211028205900346.png" alt="image-20211028205900346" style="zoom: 50%;" />

  Y5 上面的一个翻译是上下文无关 S-NMT 的翻译，但和整个句子的连贯性不足。

- Model :

  ![image-20211029105811829](https://img.wzf2000.top/image/2021/10/29/image-20211029105811829.png)

- Training Object : 

  1. 在大规模句级别NMT数据上minimize：

     ![image-20211029111328400](https://img.wzf2000.top/image/2021/10/29/image-20211029111328400.png)

  2. 在对话翻译数据上maximize：

     ![image-20211029111418172](https://img.wzf2000.top/image/2021/10/29/image-20211029111418172.png)

- 结果：

  ![image-20211029111457085](https://img.wzf2000.top/image/2021/10/29/image-20211029111457085.png)



## 35. Importance-based Neuron Allocation for Multilingual Neural Machine Translation

- Link : [Importance-based Neuron Allocation for Multilingual Neural Machine Translation](https://aclanthology.org/2021.acl-long.445.pdf)

- Affiliation : Chinese Academy of Sciences (ICT/CAS) , University of Chinese Academy of Sciences, Beijing Language and Culture University

- Authors : Wanying Xie, Yang Feng, Shuhao Gu, Dong Yu

- 摘要：Multilingual NMT 在学习 language-specific 知识上有困难。本文提出了一种方法，把模型中的 neurons 分为 general 和 language-specific 两部分，而无需加入新模块、新参数。（相似的语言中相似的特征可以被一些特定 neurons 捕捉到。）在 IWSLT 和 Europarl 数据集上的多个语言对上实验，显示了有效性和普遍性。

- Method :

  ![image-20211029113654769](https://img.wzf2000.top/image/2021/10/29/image-20211029113654769.png)

  Importance Evaluation : 

  ![image-20211029113823914](https://img.wzf2000.top/image/2021/10/29/image-20211029113823914.png)

- 结果：

  ![image-20211029113729036](https://img.wzf2000.top/image/2021/10/29/image-20211029113729036.png)



## 36. Good for Misconceived Reasons: An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation

- Link : [Good for Misconceived Reasons: An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation](https://aclanthology.org/2021.acl-long.480.pdf)

- Affiliation : The University of Hong Kong, Tencent AI Lab, Shanghai Artificial Intelligence Laboratory, East China Normal University
- Author : Zhiyong Wu , Lingpeng Kong, Wei Bi , Xiang Li , Ben Kao

- 摘要：神经多模态机器翻译（MMT）系统是一个旨在通过扩展具有多模态信息的传统纯文本翻译模型来实现更好翻译的系统。这些改进是否确实来自多模态部分存在争议。本文通过设计两个可解释的MMT模型，重新审视了MMT中多模态信息的贡献。令人惊讶的是，尽管该模型复现了最近开发的多模式集成系统所取得的类似成果，但模型学会了忽略多模态信息。经过进一步的研究，发现多模态模型相对于纯文本模型所取得的改进实际上是正则化效应（视为随机噪声，增强鲁棒性）的结果，尤其是在文本数据充足的情况下；而在文本数据有限时，图像部分可以提高其性能。本文构建的可解释 MMT 模型可以作为一个新的 baseline 。

- Model :

  - Gate Fusion MMT

    <img src="https://img.wzf2000.top/image/2021/11/03/image-20211103063222166.png" alt="image-20211103063222166" style="zoom:50%;" />

    <img src="https://img.wzf2000.top/image/2021/11/03/image-20211103063330681.png" alt="image-20211103063330681" style="zoom:50%;" />

  - Retrieval-Augmented MMT

    相似度（内积）

    text 部分使用 BERT

    <img src="https://img.wzf2000.top/image/2021/11/04/image-20211104134155451.png" alt="image-20211104134155451" style="zoom:50%;" />

- Result : 

  ![image-20211103062848871](https://img.wzf2000.top/image/2021/11/03/image-20211103062848871.png)

- Ignoring visual part : 

  <img src="https://img.wzf2000.top/image/2021/11/03/image-20211103063011601.png" alt="image-20211103063011601" style="zoom:50%;" />

   a larger gating weight Λij indicates the model learns to depend more on visual context

  <img src="https://img.wzf2000.top/image/2021/11/03/image-20211103063455082.png" alt="image-20211103063455082" style="zoom:50%;" />

  在训练中迅速下降至0

- limited textual context?

  <img src="https://img.wzf2000.top/image/2021/11/03/image-20211103063819781.png" alt="image-20211103063819781" style="zoom:50%;" />




## 37. Selective Knowledge Distillation for Neural Machine Translation

- Link : [Selective Knowledge Distillation for Neural Machine Translation](https://aclanthology.org/2021.acl-long.504.pdf)

- Affiliation : Peking University & WeChat AI

- Authors : Fusheng Wang , Jianhao Yan , Fandong Meng , Jie Zhou

- 摘要：为了提升 NMT 模型的性能，可以进行知识蒸馏。在蒸馏过程中，把 teacher model  的知识迁移到每个 training sample 上。本工作设计了一个 protocol ，能够通过比较不同例子的分区来有效分析例子的不同影响。通过实验发现，有一些教师知识并不能提高蒸馏效果，反而有负面影响。本文提出了两种筛选策略：batch-level selection 和 global-level selection 。

- 部分结果：

  <img src="https://img.wzf2000.top/image/2022/01/10/image-20220110002438562.png" alt="image-20220110002438562" style="zoom: 33%;" />



## 38. 











































## Prefix-Tuning: Optimizing Continuous Prompts for Generation

- Link : [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190#:~:text=%20%5B2101.00190%5D%20Prefix-Tuning%3A%20Optimizing%20Continuous%20Prompts%20for%20Generation,the%20language%20model%20parameters%20and%20therefore%20necessitates%20storing)

- Affiliation : Stanford

- Authors : Xiang Lisa Li , Percy Liang

- template的构建：不采用离散的template token，而使用连续可调的矩阵来调整template

- 将 prompt-tuning用于语言模型的生成任务上

  <img src="https://img.wzf2000.top/image/2021/12/14/image-20211214233718834.png" alt="image-20211214233718834" style="zoom: 50%;" />

- "Compared to this line of work, which tunes around 3.6% of the LM parameters, our method obtains a further 30x reduction in task-specific parameters, **tuning only 0.1% while maintaining comparable performance**."

- Example

  ![image-20211214234132309](https://img.wzf2000.top/image/2021/12/14/image-20211214234132309.png)
