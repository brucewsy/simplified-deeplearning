# 神经机器翻译（Neural Machine Translation）
## BLEU-机器翻译的自动评价方法
BLEU的全称为Bilingual evaluation understudy

> 参考论文（K. Papineni et al. 2002） </br>
> K. Papineni, S. Roukos, T. Ward, and W. J. Zhu. BLEU: a method for automatic evaluation of machine traslation. In ACL, 2002.

对于机器翻译，人工评价的开销很大，所以IBM提出了一种机器翻译的自动评价方法，即BLEU。
那么，我们如何评价一个翻译的好坏？在参考论文中，作者认为越接近于专业人员翻译的机器翻译，就是一个优秀的翻译。

以这个作为核心的想法，一个判断机器翻译质量的方法是，根据一个数值型指标，来评价译文与一个或多个参考译文的相近程度。所以，在这套机器翻译评价体系中，有两个前提：
1. 一个数值型的“翻译相似度”；
2. 一组高质量的翻译参考译文。

首先来看例1，包含两段待评价译文，与三段参考译文：

> 例1： </br>
> 待评价译文 1： </br>
> It is a guide to action which ensures that the military always obeys the commands of the party. </br>
> 待评价译文 2： </br>
> It is to insure the troops forever hearing the activity guidebook that party direct.
>
> 参考译文 1： </br>
> It is a guide to action that ensures that the military will forever heed Party commands. </br>
> 参考译文 2： </br>
> It is the guiding principle which guarantees the military forces always being under the command of the Party. </br>
> 参考译文 3： </br>
> It is the practical guide for the army always to heed the directions of the party.

很明显的，待评价译文1要优于待评价译文2。我们可以看到在待评价译文1中使用的单词，许多都出现在了参考译文中，而待评价译文2的精确度要低的多。我们可以通过对比n-grams来确定精确度。但是这样的情况，存在一个问题，请看例2。

> 例2： </br>
> 待评价译文 </br> 
> the the the the the the the
>
> 参考译文 1： </br>
> The cat is on the mat. </br>
> 参考译文 2： </br>
> There is a cat on the mat.

在本例中，待评价译文的精确度是100%。很明显这么分析是不对的。而实际的原因就是：参考译文中的单词，在匹配后不应重新计入匹配的状态。换而言之，参考译文中的单词，只能匹配一次，不重复匹配。从而，我们引出了**调整n-gram精确度（Modified n-gram precision）**。这类的调整n-gram精确度评分，涵盖了翻译的两部分内容：**充分性**和**流利度**。充分性体现在了一段翻译使用的单个单词（1-gram）趋于参考中文所使用的单词。流利度体现在使用较长的n-gram去匹配。

而对一段测试文本，其中的一句往往不能代表本文的翻译水平，于是需要计算整段文本翻译精确度$p_n$，公式如下：

$$p_n = \frac {\sum_{C \in \{Candidates\}} \sum_{n-gram \in C} Count_{clip}(n-gram)} {\sum_{C' \in \{Candidates\}} \sum_{n-gram' \in C'} Count(n-gram')}$$

首先，一句一句计算n-gram的匹配结果；接着，对所有的候选译文累加修剪后的n-gram数量，再除以在测试文本中候选n-gram的数量。即可得到调整后的精确度$p_n$。

由实验结果表明，当$n=4$的时候，有很最好的相关性（correlation），相比3-grams和5-grams，则给出了有比较性的结果。

对于候选翻译的长度不一这个特点，评价指标应该将它也考虑其中。而精确度$p_n$不能满足这个要求，见例3：

> 例3： </br>
> 待评价译文：</br>
> of the
>
> 参考译文 1： </br>
> It is a guide to action that ensures that the military will forever heed Party commands. </br>
> 参考译文 2： </br>
> It is the guiding principle which guarantees the military forces always being under  the command of the Party. </br>
> 参考译文 3： </br>
> It is the practical guide for the army always to heed the directions of the party.

在例3中，我们的调整unigram精确度为$2/2=1$；调整bigram精确度为$1/1 = 1$。

与精确度相对的是召回率（recall）。然而，BELU使用多段参考译文，而在每一个参考译文中，一个单词可能会有不用的翻译单词和他对应。更深的说，一个好的待评价译文仅仅使用一个参考译文，而不是所有的参考译文。如例4：

> 例4： </br>
> 待评价译文 1： </br>
> I always invariably perpetually do. </br>
> 待评价译文 2： </br>
> I always do.
>
> 参考译文 1： </br>
> I always do. </br>
> 参考译文 2： </br>
> I invariably do. </br>
> 参考译文 3： </br>
> I perpetually do. </br>

在待评价译文1中，尽可能的使用了参考译文中出现的单词，与待评价译文2相比，这显然不是一个好的译文。所以用简单的召回率计算所有的参考译文中出现的单词，显然不是一个好的方法。

为了克服这个问题引进了一个**惩罚因子（brevity penalty）**，希望当候选译文的长度等于某一参考译文的长度，则惩罚因子为1.0。例如：如果有三个参考译文分别有12、15、17个单词，候选译文有12个单词，此时我们希望惩罚因子为1.0。
另一个问题是，如果我们一句一句的计算惩罚因子，然后取平均，长度较短的句子将被惩罚的更严重。所以，选择通过整个文本计算惩罚因子，并且允许在句子上有一定的自由度。

首先计算测试文本有效参考长度$r$，即对文本中每个候选译文的最优匹配长度求和。接着，通过以$e$为底、$r/c$为指数的指数级衰减，这里$c$是候选（待评价）译文文本的总体长度。

$$BP = 
\begin{cases}
1                 & , if \quad c \gt r \\
e^{1-{r \over c}} & , if \quad c \leq r 
\end{cases}
$$

通过计算$BP$与$p_n$，可以得到BLEU值：

$$BLEU = BP * exp(\sum_{n=1}^{N} W_n \log p_n)$$

也可以同时两边取$\log$，得到：

$$\log BLEU = min(1-{r \over c}, 0) + \sum_{n=1}^{N} W_n \log p_n$$

需要说明的是，BELU的值介于0.0至1.0之间。很少有翻译可以达到1.0的评价，除非他们与参考译文是一致的。基于这些理由，就算是人工翻译的译文也不一定可以达到1.0的评价。

------------------------------------------------------------------

## 神经机器翻译

> 论文参考： </br>
> Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. ICLR. </br>
> Minh-Thang Luong, Hieu Pham, and Christopher D Manning. 2015. Effective approaches to attention-based neural machine translation. EMNLP.

### 基础的神经机器翻译
&emsp;&emsp;首先与传统的机器翻译做个对比。传统的机器翻译是基于短语的机器翻译；同时有许多调整过的子成分组成的。

神经机器翻译，则尝试建立训练一个单一的、庞大的神经网络，他由编码器-解码器（encoder-decoder）组成。可以说他是编码器-解码器家族的一员。

神经机器翻译，通常使用的是RNN结构(Luong et al. 2015):
只是对于编码-解码器，会使用不同的RNN形式：
- Kalchbrenner and Blunsom(2013)
    - 编码器：使用CNN
    - 解码器：使用标准隐藏单元的RNN
- Sutskever et al.(2014)和Luong et al.(2015)
    - 编码器、解码器：使用LSTM的隐藏单元的RNN
- Cho et al.(2014), Bahdanau et al.(2015)和Jean et al.(2015)
    - 采用了启发式LSTM隐藏单元(LSTM-inspired hidden unit)与gated recurrent unity(GRU)混合形式的RNN

### 普通的RNN编码器-解码器
在编码器-解码器结构中，编码器（encoder）读取一个输入语句，即一个向量序列$x=(x_1,  ...,  x_{T_x})$，将其转化成一个向量$c$。最常用的方式就是RNN

$$h_t = f(x_t,  h_{t-1})$$

和

$$c = q({h_1,   ...,   h_{T_x}})$$

其中$h_t \in R^n$是$t$时刻的隐藏状态，$c$是一个从隐藏状态序列中生成的向量，$f$与$g$是一些非线性方程。

解码器（decoder）通常被用来训练，在给定上下文向量$c_t$与所有之前预测的单词$\{y_1, ..., y_{t'-1}\}$的情况下，预测下一个单词$y_{t'}$。换而言之，解码器定义了在

$$
p(y) = \prod_{t=1}^T p(y_t   |   \{y_1,   ...,   t_{t-1}\},   c)
$$

这里$\rm {y} = (y_1, ..., y_{T_y})$。对RNN，每个条件概率模型为：

$$p(y_t | \{y_1, ..., y_{t-1}\}, c) = g(y_{t-1}, s_t, c)$$

其中$g$为输出$y_t$概率的非线性、多层方程，$s_t$是RNN的隐藏层。


### 对齐与翻译
Bahdanau D. et al(2015)提出了一种改进方式，使用两双向RNN作为编码器，在解码过程中，仿真搜索源输入。

解码器

新模型中，重新定义了条件概率：

$$p(y_t | \{y_1, ..., y_{t-1}\}, \rm {x}) = g(y_{t-1}, s_t, c_i)$$

其中，$s_i$是RNN在$t$时刻的隐藏状态，计算如下：

$$s_i = f(s_{i-1},   y_{i-1},   c_i)$$

这里需要注意的就是，不同于之前的公式，这里对于每个目标单词$y_i$使用了不同的本文向量$c_i$。

上下文向量$c_i$通过这些$h_i$的权重相加求得：

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

这里的$\alpha_{ij}$通过每个$h_j$得到：

$$\alpha_{ij} = \frac {\exp(e_{ij})} {\sum_{k=1}^{T_x} \exp(e_{ik})}$$

其中

$$e_{ij} = a(s_i-1, h_j)$$

这个一个对齐模型（alignment model），计算了在$j$位置的输入与$i$位置的输出的匹配程度。这个评价是基于RNN的隐藏状态$s_{i-1}$与输入语句的第$j$个注解$h_j$。

参数化对齐模型$a$将它视为反馈神经网络（feedforward neural network），它联合系统其他部分一同训练。这有别于传统的机器翻译，对齐不被认为是隐式因子。相反，对齐模型直接计算了软对齐（soft alignment），这允许代价函数的梯度反向传播。这个梯度可以被用来训练对齐模型与整个翻译模型。

编码器

通常RNN从第一个字符$x_1$到最后一个字符$x_{T_x}$有序的读取序列$\rm x$。这里作者做了调整，希望，每个单词的注解（annotation）不仅包括单词本身，还希望它包括后续单词。所以这里使用了双向RNN（bidirectional RNN）。

BiRNN包括前馈RNN与反馈RNN。前馈RNN$\overrightarrow{f}$有序的读取序列，计算出前馈隐藏状态
$$(\overrightarrow{h}_1, ..., \overrightarrow{h}_{T_x})$$

的序列。反馈RNN$\overleftarrow{f}$反向读取序列，生成一个反向隐藏状态$(\overleftarrow{h}_1, ..., \overleftarrow{h}_{T_x})$的序列。

对每个单词$x_j$的注释（annotation），可以通过组合前馈隐藏状态$\overrightarrow{f}$和反馈隐藏状态$\overleftarrow{f}$得到，即$h_j = [\overrightarrow{h}_t^T; \overleftarrow{h}_t^T]^T$。这样注解$h_j$同时包含了预测单词与后续单词的信息。由于RNN对当前的输入会有更好的表示，注解$h_j$将会关注于单词$x_j$附近的信息。

### 注意力机制（attention machenism）

$$\log p(y | x) = \sum_{j=1}^{m} \log p(y_j | y_{\lt j}, s)$$

$$p(y_j | y_{\lt j}, s) = softmax(g(h_j))$$

$$h_j = f(h_{j-1}, s)$$

目标函数为：

$$J_t = \sum_{(x,y)\in D} - \log p(y | x)$$

Luong M.T. et al.(2016)提出了另一种模式，引入了全局注意力（global attetntion）与局部注意力（local attention）。其实就是在解码阶段，每步t时刻，两个方法都是先获取LSTM在顶层的隐藏状态$h_t$。接着就是导出上下文向量（context vector）$c_t$，它包含了相关输入端（source-side）的信息，来帮助预测当前目标单词$y_t$。而这两个模型不同的就是上下文向量$c_t$的导出方式。

接着，给定目标隐藏状态（target hiddent state）$h_t$与输入端的上下文向量（source-side context vector）$c_t$，结合两个向量生成注意力的隐藏状态（attentional hidden state）：

$$\tilde{h}_t = \tanh (W_c [c_t; h_t])$$

而后，再将$\tilde{h}_t$通过$softmax$层，生成预测分布方程：

$$p(y_t | y_{\lt t}, x) = softmax(W_s \tilde{h}_t)$$

#### 全局注意力（global attention）
全局注意力（gloabl attention）的核心就是，在导出上下文向量$c_t$的时候，考虑编码器的所有隐藏状态。在这个模型类型中，边长对齐向量$a_t$是由对比当前目标隐藏状态$h_t$和每个源隐藏状态（source hidden state）$\tilde{h}_t$:

$$
a_{t}(s) = align(h_t, \bar {h}_s) = \frac {\exp(score(h_t, \bar{h}_s))} {\sum_{s'}\exp(score(h_t, \bar{h}_{s'}))}
$$


这里，$socre$根据基于内容的方程（content-base function）推出，有三中选择方式：

$$
score(h_s, \bar{h}_t) = \begin{cases}
h_t^T \bar{h}_s                  & dot\\
h_t^T W_a \bar{h}_s              & general\\
v_a^T \tanh (W_a[h_t; \bar{h}_s]) & concat
\end{cases}$$

相比Bahdanau et al.(2015)，注意力机制都很相似，但有几个关键点不同：
全局注意力简化了使用在顶层LSTM的隐藏状态；
全局的计算路径更为简单：$h_t$ -> $a_t$ -> $c_t$ -> $\tilde{h_t}$，
而前者不是$h_{t-1}$ -> $a_t$ -> $c_t$ -> $h_t$；

局部注意力（loacl attention）

由于全局注意力有个弊端，它需要对每个目标单词，都要在输入端注意所有的单词，这样的开销是巨大的，特别是当输入语句很长时。为了克服这个问题，作者提出了局部注意力机制（local attention mechanism）

局部注意力机制有选择性的关注于内容的有个小窗口，这个方法避免了计算的开销，并且更容易训练。

首先，模型在$t$时刻，对于每个目标单词，生成一个对齐的位置$p_t$。接着，通过在窗口$[p_t-D, p_t+D]$上加权，生成上下文向量$c_t$，要说明的是$D$是经验选择。不像全局注意力机制，局部对齐向量$a_t$现在是一个固定维度的向量，$\in R^{2D+1}$。这里将这个模型分为两种：

单调对齐（Monotonic alignment，loacl-m），简单的设置$p_t=t$，假设输入与目标序列大致单调对齐。这样的对齐向量$a_t$可以用过函数（）定义。

预测对齐（predictive alignment，local-p），替换单调对齐，模型预测一个对齐位置：

$$p_t = S \cdot sigmode(v_p^T \tanh (W_p h_t))$$

其中$W_p$和$v_p$是模型参数，被用来学习预测位置。$S$是源输入的长度。在$sigmod$之后，$p_t \in [0,S]$。

为了偏向$p_t$附近的对齐点，在以$p_t$为中心，放置一个高斯分布。特别的，定义对齐权重为：

$$a_t(s) = align(h_t, \bar {h}_s) \exp (- \frac {(s-p_t)^2} {2 \sigma ^2})$$

使用和函数（）相同的对齐方程（align fucntion），同时标准差是通过经验设置为$\sigma = \frac {D} {2}$。这里$p_t$是实数，相比$s$是在以$p_T$为中心窗口下的整数。


$$$$

