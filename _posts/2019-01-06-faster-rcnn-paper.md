---
layout: post
title:  "Faster R-CNN论文的中文翻译（不含图片表格）"
date:   2019-01-06 15:24:44 +0800
categories: ml
tags: cnn computer-vision object-detection
mathjax: true
---

* content
{:toc}

Markdown format, $\TeX$ required

IEEE Transaction on Pattern Recognition and Machine Intelligence  
(Volume: 39, No: 6, Jun 2017, pp. 1137-1149)  
[https://ieeexplore.ieee.org/document/7485869](https://ieeexplore.ieee.org/document/7485869)

---


# Faster-RCNN: 用区域候选网络实现实时物体检测

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun





## 摘要

现有的对象检测网络依赖于区域候选算法来假设物体的定位。就如SPPnet和Fast R-CNN这样的优化方法减少了这些检测网络的运行时间，暴露出区域候选计算成为瓶颈的问题。在这项工作中，我们引进了*区域候选网络*（Region Proposal Networks, RPN）来共享检测网络的全图像卷积特征，由此来实现近乎没有成本的区域候选。RPN是一个可以同时预测每个位置上的物体边界和物体得分的全卷积网络。RPN采用端对端训练来生成高质量的候选区域，并且用于Fast R-CNN的检测。我们通过使用最近一个热门的神经网络术语——“注意（attention）”机制，来共享卷积特征，进一步整合了RPN和Fast R-CNN成一个单一的网络。RPN组件告诉统一的网络哪里需要观察。对于VGG-16模型，我们的检测系统在GPU上能有5fps的帧速度（*包括所有步骤*），并且在每个图片只有300个候选（proposals）结果时，在PASCAL VOC2007、2012和MS COCO数据集中均取得了现有的物体检测的最高精确度。在ILSVRC和COCO 2015比赛中，Faster R-CNN和RPN是多个赛道上第一名参赛作品中的基础部分。代码已经公开发布。

**索引项**——物体检测、区域候选、卷积神经网络

## 1. 引言

最近在物体检测上的进步都取决于区域候选方法（如[4]）的成功和基于区域的卷积神经网络（R-CNN）的方式[5]。尽管基于区域的CNN计算开销都很大，如最初开发的[5]，但是他们的开销能够通过在[1]、[2]中在生成候选结果时共享卷积（特征）来大幅减小。在最新的研究Fast R-CNN中，它通过深度网络的方法，*在忽略了区域候选上的时间开销后*，取得了几乎实时性的成果。现在，区域候选是现有检测系统中，进行测试时的计算瓶颈。

区域候选方法一般依赖于低成本的特征和经济的推理方案。选择性搜索（Selective Search），最热门的方法之一，在基于低等特征工程之上贪婪地合并超像素（superpixel）。然而，跟高效的检测网络相比，选择性搜索的速度慢了一个数量级，在CPU的实现中的速度为2秒一张图片。EdgeBoxes [6]目前提供了候选质量和速度上的最佳平衡，大约为0.2秒一张图片。尽管区域候选步骤依然消耗跟检测网络一样长的运行时间。

你们可能注意到，快速的基于区域的CNN运用了GPU的优势，而研究中的区域候选方法是用CPU实现的，使得在运行时的对比会有失公平。一个明显的加速候选计算的方法就是在GPU上重新实现它。这可能是一个有效的工程解决方案，但是重新实现忽略了下游的检测网络，从而失去了共享计算的重要机会。

在这篇论文中，我们展示了算法上的改变——使用深度卷积网络计算候选区域——引出一种优雅高效的解决方案，在这方案中，对于给定一个检测网络而言，候选计算几乎是没有开销的。为此，我们引入了一个新名词*区域候选网络*（RPNs），它可以共享现有的物体检测网络中的卷积层[1]，[2]。通过在测试时共享卷积操作，计算候选区域的边缘开销变得很小（如每张图片10ms）。

我们观察得到，卷积的特征图，用于像Fast R-CNN这样的基于区域的检测器，也可以被用来生成区域候选。在这些卷积特征的顶部，我们通过添加额外的卷积层，构建了一个RPN，这些卷积层可以在一般的网格上的所有位置同时进行区域边界的回归和物体得分。因此，RPN是一种全卷积网络（fully convolutional network, FCN） [7]，并且可以在生成候选结果的任务上进行端对端的训练。

RPN被设计成通过各种面积和宽高比例进行高效地预测候选区域。对比于流行的方法[1]，[2]，[8]，[9]而言，它们使用的是金字塔形的图片（图1，a）或者是金字塔形的过滤器（图1，b），我们引入一个新名词“锚点（anchor）”边界来作为多个面积和宽高比的参考边界。我们的方案可以被认为是一种金字塔形的回归参考（图1，c），它避免了穷举整个图片或是多种面积和宽高比的过滤器。这个模型在使用单一大小的图片进行训练和测试的情况下表现得很好，运行速度也因此得到提高。

为了统一RPN和物体检测网络Fast R-CNN，我们提出一个训练方案，它在微调区域候选任务和在保持候选结果固定下的物体检测之间交替进行。这种方案很快收敛，并且产生了一个可以在两个任务之间共享卷积特征的统一网络。

我们在PASCAL VOC检测基准[11]上综合评估了我们的方法，这种通过RPN和Fast R-CNN生成检测的方法比起使用选择性搜索的Fast R-CNN方法，在基线（baseline）上取得更高的准确度。同时，我们的方法在测试时抛弃了选择性搜索的大部分计算负担——用于候选的有效运行时间只为10毫秒。使用[3]中的大型深层模型，我们的检测方法在GPU上依然有一个5fps的帧率（*包括所有步骤*），并且对于速度和精度而言，该物体检测系统有着很好的实用型。我们也会汇报在MS COCO数据集上的结果[12]，和研究在使用COCO数据后，该模型在PASCAL VOC上的提升。代码已经公开在https://github.com/shaoqingren/faster_rcnn（MATLAB）和https://github.com/rbgirshick/py-faster-rcnn（python）。

这个手稿的原始版本在先前已经发表[10]。在此之后，RPN和Faster R-CNN的框架已经被采用并泛化为其他方法，如3D物体识别[13]，基于部分（part-based）的检测[14]，实例分段（instance segmentation）[15]，和图片拟题（image captioning）[16]。我们快速高效的物体检测系统也被用于商业系统，如Pinterests [17]，报告称用户参与度得到提升。

在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet定位，COCO检测，和COCO分段赛道上第一名参赛作品[18]中的基础部分。RPN完全是为了从数据中学习如何候选区域，并且可以容易地在更深和更有表达性的特征中得益（如在[18]中采用101层的残差网络）。Faster R-CNN和RPN也被用于在这些竞赛中的其它领先参赛作品。这些结果表明我们的方法不仅是一个实用有效率的解决方案，而且还是一个提高物体检测精度的有效方法。

## 2. 相关工作

**物体候选**。在物体候选方法上有着许多文献。物体候选方法的综合调查和对比记述在[19]，[20]，[21]中。广泛使用的物体候选方法包括那些基于超像素的方法（如选择性搜索[4]，CPMC [22]，MCG [23]）和那些基于滑动窗口的方法（如，objectness in windows [24]，EdgeBoxes [6]）。其他的候选方法是作为检测器的一个外部独立模块来使用的（如选择性搜索[4]的物体检测器，R-CNN [5]和Fast R-CNN [2]）。

**用于物体检测的深度网络**。R-CNN方法[5]使用端对端的CNN进行训练，用于分类候选区域的物体类别或是背景（即该区域没有物体）。R-CNN主要充当一个分类器，它不预测物体边界（除了通过边界回归进行修正）。它的精确性取决于区域候选模块的表现（见[20]中的比较）。几篇论文已经提出使用深度网络来预测物体边界的方法[9]，[25]，[26]，[27]。在OverFeat方法中[9]，一个全连接层被用于训练定位任务，在假设只有一个物体时预测其边界坐标。全连接层随后被替换成一个卷积层来检测多个具体类别的物体。MultiBox方法[26]，[27]从网络中生成区域候选，该网络的最后一个全连接层同时预测多个类别无关的边界，泛化了OverFeat中的“单一边界”的方法。这些类别无关的边界用于R-CNN的候选当中[5]。跟我们的全卷积方案相比，MultiBox候选网络被应用在裁剪过的单一图片的或多个图片上（如，$244\times244$）。MultiBox并没有共享候选网络和检测网络中的特征。我们在后文中会将我们的方法结合OverFeat、MultiBox等方法进行更深入的说明。在我们的工作同时，DeepMask [28]方法被开发出来用于学习分段（segmentation）候选。

卷积的共享计算[1]，[2]，[7]，[9]，[29]已经越发引起在效率，精度以及视觉识别方面上的注意。OverFeat的论文[9]从一个图片金字塔（image pyramid）中计算用于分类，定位和检测的特征。在共享卷积特征图像上的自适应大小池化（Adaptively-sized pooling，SPP）[1]被用于有效的基于区域的物体检测[1]，[30]和语义分段[29]。Fast R-CNN [2]允许在共享卷积特征上进行端对端的检测器训练，并且取得了引人注目的精度和速度。

## 3. Faster R-CNN

我们的物体识别系统称为Faster R-CNN，由两个模块组成。第一个模块是一个深度全卷积网络，用于候选区域，然后第二给模块是使用候选区域的Fast R-CNN检测器。整个系统是一个用于物体检测的统一系统（图2）。使用最近热门的神经网络术语“注意”机制，RPN模块告诉Fast R-CNN模块哪里需要观察。在3.1部分我们介绍用于区域候选的网络的设计及其属性。在3.2部分我们研究了用于共享特征模块的训练方法。

### 3.1 区域候选网络

区域候选网络（RPN）将（任意大小的）图片作为输入，并输出一个矩形的物体候选集合，每个候选结果附带一个物体得分。我们将这个过程用一个全卷积网络进行建模[7]，该过程将在这里叙述。因为我们的最终目标是通过Fast R-CNN物体检测网络进行共享计算[2]，所以我们假设两个网络对一般的卷积层都进行共享。在我们的实验中，我们研究了Zeiler和Fergus模型[33]（ZF），它有5个可共享的卷积层，我们还研究了Simonyan和Zisserman的模型[3]（VGG-16），它有13个可共享的卷积层。

为了生成候选区域，我们在最后一个共享的卷积层输出的卷积特征图上滑动一个小型网络。这个小型网络将卷积特征图上的$n\times n$窗口作为输入。每个滑动窗口映射到一个低维度的特征（ZF为256维，VGG为512维，后面附带一个ReLU [33]）。这个特征被传入到两个全连接层中——一个边界回归层（*reg*）和一个边界分类层（*cls*）。我们在这篇论文中使用$n=3$，因为注意到输入图像上的有效感知视野非常大（ZF为171像素，VGG为228像素）。这个迷你的网络如图3（左边）的一个单独的位置所示。要注意的是因为这网络是使用滑动窗口的形式进行的，所以在所有空间位置上，全连接层（的权重）都是共享的。这个架构自然通过一个$n\times n$的卷积层来实现，并且紧接着两个$1\times1$的卷积层（进行*回归*和*分类*）。

#### 3.1.1 锚点

在每个滑动窗口的位置，我们同时预测多个候选区域，其数量取决于每个位置上最大的候选数量，记为$k$。所以*回归*层有$4k$个输出，对$k$个边界进行编码，并且*分类*层有$2k$个得分，估计每个候选是否为一个物体的可能性。$k$个候选是*相对于*$k$个参考边界的参数，我们将这样的参考边界称为*锚点*（anchor）。锚点的中心位于所讨论的滑动窗口的中心，并关联一个面积和宽高比（图3，左部）。默认我们使用3种面积和3个宽高比，在每个位置生成$k=9$个锚点。对于一个有$W\times H$大小的卷积特征图而言，总共有$WHk$个锚点。

**锚点的平移不变性**。我们的方法中一个重要的属性就是*平移不变性*，无论是锚点还是相对于锚点来计算候选的函数都具有该属性。如果某人在一张图片中平移一个物体，那么候选也应该进行平移，而且同样的函数应该能够在两个位置都能预测候选。这种平移不变性通过我们的方法得以保证。作为对比，MultiBox方法[27]使用k-means来生成800个锚点，它们*不*具备平移不变性。所以MultiBox不保证在物体平移之后能够生成同样的候选。

平移不变性同时减小了模型的大小。在$k=9$个锚点的情况下，MultiBox有一个$(4+1)\times800$维度的全连接输出层，而我们的方法有一个$(4+2)\times9$维度的卷积输出层。作为结果，我们的输出层有$2.8\times10^4$个参数（VGG-16的参数为$512\times(4+2)\times9$），比MultiBox的输出层少两个数量级，MultiBox的输出层参数个数为$6.1\times10^6$（在MultiBox中，GoogleNet的参数个数为$1536\times(4+1)\times800$）。将特征投影层也考虑进来的话，我们的候选层依然比起MultiBox少了一个数量级的参数个数。就如PASCAL VOC这样的训练集，我们希望我们的方法在小的训练集上能够减少过拟合的风险。

**多面积锚点作为回归参考点**。我们设计锚点时提出一种解决多面积（和多宽高比）的新方案。如图1所示，一共有两种流行的方法用于多面积的预测。第一种方法是基于图像/特征金字塔，如，在DPM[8]中和基于CNN的方法[1]，[2]，[9]。图片被缩放到多个面积，而且对每个面积计算对应的特征图（HOG [8]或深度卷积特征[1]，[2]，[9]）（图1（a））。这种方法通常很实用，但是很花费时间。第二种方法是在特征图上使用多个面积（和/或多种宽高比）的滑动窗口。例如在DPM [8]中，不同宽高比的模型使用不同的过滤器大小进行单独训练（例如$5\times7$和$7\times5$）。如果这种方法用来解决多面积问题，那么它可以被认为是一种“过滤器金字塔”的方法。第二种方法通常包括了第一种方法[8]。

作为比较，我们基于锚点的方法是建设在*锚点金字塔*之上的，这种方法会更有效率。我们的方法通过参考多种面积和宽高比的锚点边界，进行边界分类和回归。它只依赖于单一大小的图片和特征图，并且使用单一大小的过滤器（在特征图上的滑动窗口）。我们通过实验展示这种多面积和宽高比方案的效果。

#### 3.1.2 损失函数

为了训练RPN，我们为每个锚点赋予了一个二元分类标签（是否为一个物体）。我们对两种类型的锚点赋予了正标签：（1）跟ground-truth的边界拥有最高交并比（Intersection-over-Union，IoU）的一个或多个锚点，*或者*是（2）存在一个ground-truth的边界，使其边界和锚点边界有超过0.7的IoU比例的锚点。注意的是一个ground-truth的边界可以将正标签赋值给多个锚点。通常情况下，第二种条件足够确定正样本了；但是我们依然采用第一种条件，因为在某些罕见的情况下，第二种条件会找不到正样本。在剩下的锚点中，我们将对所有ground-truth的边界的IoU比例低于0.3的锚点赋予负标签。没有赋予任何正负标签的锚点不参与训练。

通过这些定义，我们会在接下来的Fast R-CNN的多任务损失中，最小化目标函数[2]。我们对一个图片的损失函数定义为：

$$\begin{aligned}
L(\{p_i\},\{t_i\}) &= \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) \\
&+\lambda\frac{1}{N_{reg}}\sum_i p_i^*L_{reg}(t_i, t_i^*)
\end{aligned}\tag{1}$$

在这里，$i$是在一个mini-batch中的锚点的索引，$p_i$是预测出的锚点$i$是一个物体的可能性。ground-truth的标签$p_i^* $在锚点为正锚点时为$1$，负锚点时为$0$。$t_i$是一个代表预测边界的4个坐标参数的向量。$t_i^* $是正锚点相关联的ground-truth的边界。分类损失$L_{cls}$是两个类别（是一个物体vs.不是一个物体）的对数损失。对于回归损失，我们使用$L_{reg}(t_i,t_i^* )=R(t_i-t_i^* )$，其中$R$是定义在[2]中的robust损失函数（Smooth $L_1$）。项$p_i^* L_{reg}$意思是只在正锚点（$p_i^* =1$）时激活回归损失，在其余情况下（$p_i^* =0$）则忽略该项。*分类*和*回归*层的输出分别为$$\{p_i\}$$和$$\{t_i\}$$。

这两项通过$N_{cls}$和$N_{reg}$进行归一化，并且由一个均衡参数$\lambda$进行加权求和。在我们目前的实现中（如在已经发布的代码中所示），在等式（1）中的*分类*项使用mini-batch的大小进行归一化（即，$N_{cls}=256$），且在*回归*项通过锚点的数量进行归一化（即，$N_{reg}\sim2,400$）。默认情况下，我们让$\lambda=10$，使得*分类*项和*回归*项能粗略地得到相等的权重。我们通过实验表明，实验结果对于大范围的$\lambda$而言是不敏感的（表9）。我们也注意到上述的归一化项是不需要的，并且能够被进一步简化。

对于区域边界回归，我们采用如下所示的4个坐标参数[5]：

$$\begin{aligned}
t_\text{x} &= (x-x_a)/w_a, \quad t_\text{y}=(y-y_a)/h_a, \\
t_\text{w} &= \log(w/w_a), \quad t_\text{h}=\log(h/h_a), \\
t_\text{x}^* &= (x^*-x_a)/w_a, \quad t_\text{y}^*=(y^*-y_a)/h_a, \\
t_\text{w}^* &= \log(w^*/w_a), \quad t_\text{h}^*=\log(h^*/h_a),
\end{aligned}
\tag{2}$$

其中，$x$，$y$，$w$，和$h$代表了边界的中心坐标和它的宽高。变量$x$，$x_a$，和$x^*$分别为预测边界，锚点边界和ground-truth边界（对$y$，$w$，$h$同理）。这可以被认为是一种从锚点边界到相邻ground-truth边界的边界回归。

虽然我们和前面基于RoI（Region of Interest）的方法[1]，[2]相比而言，以一种不同的形式实现了区域边界回归。在[1]，[2]中，边界回归是通过来自*任意*大小的RoI的特征并将其池化来实现的，并且回归权重在所有区域大小上是*共享*的。在我们的公式中，用于回归的特征都是*相同*空间大小的，在特征图上的大小为$(3\times3)$。若考虑到不同大小的情况，则需要训练$k$个边界回归器。每个回归器只对一种面积和一种宽高比负责，并且$k$个回归器*不*共享权重。因此，得益于锚点的设计，尽管特征都被固定在一个大小/面积，它依然能够预测不同大小的边界。

#### 3.1.3 训练RPN

RPN可以通过反向传播和随机梯度下降（SGD）进行端对端的训练[35]。我们也跟随着[2]中的以“图片为中心”的采样策略训练这个网络。每个mini-batch从一个图片中产生可能包含许多正负样本的锚点。我们可以对所有锚点都进行损失函数的优化，但是这会偏向于负样本，因其占据了主导地位。相反的，我们在一个mini-batch上随机采样256个锚点来计算损失函数，其中采样到的正负锚点*最多*能达到1:1的比例。如果一个图片有少于128个正样本，我们会使用负样本进行填充。

我们通过零均值，标准差为0.01的高斯分布来随机地初始化所有新层的权重。对于ImageNet分类而言，其他层（即，共享的卷积层）使用预训练的模型来初始化[36]，并且将其作为标准做法[5]。我们调整了ZF net，conv3_1以及VGG net的所有层以节约内存[2]。在PASCAL VOC数据集上，我们对6万次mini-batch使用0.001的学习率，并且在接下来的2万次mini-batch使用0.0001的学习率。我们使用一个0.9的动量和一个0.0005的衰减率[37]。我们的实现（代码）使用的是Caffe（框架）[38]。

### 3.2 共享RPN和Fast R-CNN的权重

到目前为止，我们已经描述了如何训练网络进行区域候选的生成，但没有考虑到使用这些候选的基于区域的物体检测CNN。对于检测网络，我们采用了Fast R-CNN [2]。接下来我们描述将共享卷积层的RPN和Fast R-CNN组合成一个统一网络的学习算法。

对RPN和Fast R-CNN单独进行训练将会以不同的方式改变其卷积层。因此我们需要开发一种技术，使得在两个网络之间能够共享卷积层，而不是学习两个分离的网络。我们讨论三种用于共享特征的训练网络的方法：

（1）*交替训练*。在该方案中，我们首先训练RPN，然后使用候选来训练Fast R-CNN。然后通过Fast R-CNN调整过的网络来初始化RPN，并且迭代该过程。这种方法是本论文中所有实验所采用的方案。

（2）*近似联合训练*。在该方案中，RPN和Fast R-CNN网络在训练时合并到一个单一网络，如图2所示。在每次SGD迭代中，前向传递生成区域候选，并且在训练Fast R-CNN检测器时将其视为固定的，并且是通过预计算得出的候选。反向传播则正常进行，对于共享层，来自RPN和Fast R-CNN的损失信号被合并在一起。这个方案很容易实现。但是这个方案忽略了关于候选边界坐标的导数，因其坐标也是网络的响应，所以它是近似的。在我们的实验中，我们凭实际经验发现这个求解器生成了相近的结果，并且跟交替训练比起来节省了大约25-50%的训练时间。这个求解器已经包含在发布的python代码中。

（3）*非近似联合训练*。如上面讨论所说，通过RPN预测的边界同样也是用于输入的函数。在Fast R-CNN中的RoI池化层[2]将这卷积特征和预测的边界作为输入，所以一个理论上合理的反向传播求解器应该也要将关于边界坐标的梯度包含进来。这些梯度在上面的近似联合训练中被忽略了。在非近似联合训练方案中，我们需要一个关于边界坐标，并且可微的RoI池化层。这是一个非平凡（nontrivial）的问题，并且可以通过在[15]中开发的“RoI warping”层给出解决方案，这超出了本论文的范围。

**4步交替训练**。在本论文中，我们采用的是实用的4步训练算法，通过交替优化从而学习共享特征。在第一步中，我们如3.1.3部分所描述的方式来训练RPN。该网络使用ImageNet的预训练模型进行初始化，并且对区域候选任务进行端对端的微调。在第二步中，我们使用第1步中RPN生成的区域候选来通过Fast R-CNN训练一个单独的检测网络。这检测网络同样用预训练的ImageNet模型进行初始化。在这个点上，两个网络并没有共享卷积层。在第三步中，我们使用检测器网络来初始化RPN训练，但是我们固定了共享的卷积层，只微调属于RPN的层。现在，两个网络共享了卷积层。最后，保持共享的卷积层不变，我们微调只属于Fast R-CNN的层。因此，两个网络都共享卷积层，并且形成的一个统一网络。一个相似的交替训练能被运行更多次的迭代，但我们观察到的只有可忽略的提升。

### 3.3 实现细节

我们在单一大小的图片上训练并测试了区域候选和物体识别网络[1]，[2]。我们重新缩放了图片，使得他们的最短边为$s=600$像素[2]。多面积特征提取（使用图片金字塔）可能会提高精度，但没有展现出好的速度-精度权衡[2]。在重新缩放的图片上，ZF和VGG net在最后的卷积层上的步幅（stride）都为16像素，因此在调整大小之前（$\sim500\times375$），一般的PASCAL特征图都为$\sim10$像素。即使是这样大的步幅也能提供良好的结果，使用更小的步幅可能会进一步提高精度。

对于锚点，我们使用3种边界面积为$128^2$，$256^2$，$512^2$像素，和3种宽高比为1:1，1:2，和2:1的锚点。这些超参数并*没有*在特定的数据集中仔细地选择过，并且我们会在下个部分提供关于其效果的剥离实验（ablation experiment）。正如所讨论的情况，我们的方案不需要图片或是过滤器金字塔来预测多面积的区域，节省了大量的运行时间。图3（右）展示了我们的方法能够适用于大范围的面积和宽高比的能力。表1展示了使用ZF net下学到的每个锚点的平均候选大小。我们注意到我们的算法允许预测大于底层的感知视野。这样的预测不是不可能的——在只有物体的中间可见的情况下，人们仍然能粗略推断出物体的范围。

超出图片边界的锚点边界需要仔细地处理。在训练过程中，我们忽略了所有超出边界的锚点，所以它们对损失没有贡献。对于一个典型的$1000\times600$的图像，总共有大概20000（$\approx\space60\times40\times9$）个锚点。在忽略了超出图片边界的锚点之后，每个图片仍然有6000个锚点能被用于训练。如果在训练时未忽略超出边界的外部锚点，那么它们就会在目标中引入一个数量大且难以纠正的误差项，并且训练不会收敛。然而，在测试时，我们依然将全卷积RPN应用在整个图片中。这可能会生成超出图片边界的候选区域，我们将其裁剪到图片边界之内。

一些RPN候选会互相高度重叠。为了减少冗余，我们根据其*分类*得分，在候选区域上采用了非极大值抑制（NMS）。我们固定NMS的IoU阈值为0.7，它可以在每个图像上保留约2000个候选区域。如我们将在后面展现的一样，NMS并不会损害最终的检测精度，但能基本上减少候选的数量。在NMS之后，我们使用排名前N的候选区域进行检测。接下来，我们使用2000个RPN候选来训练Fast R-CNN，但是在测试时使用不同数量的候选进行评估。

## 4. 实验

### 4.1 在PASCAL VOC上的实验

我们在PSCAL VOC 2007的检测基准上综合评估了我们的方法[11]。该数据集包含了5千张用于训练验证的图片和5千张用于测试的图片，总共包括了20个类别。我们同时提供了少量模型在PASCAL VOC 2012上的基准结果。对于预训练的ImageNet网络，我们使用“快速”模式的ZF net [32]，它包含了5个卷积层和3个全连接层，公开的VGG-16模型包含了13个卷积层和3个全连接层。我们主要评估了检测的平均精度均值（mean Average Precision，mAP），因为这是用于物体检测上的实际指标（比起专注于物体候选代理指标（proxy metric）而言）。

表2（顶部）展示了Fast R-CNN使用不同区域候选方法训练和测试的结果。这些结果均使用了ZF net。对于选择性搜索（SS）[4]，我们在“快速”模式下生成了大约2000个候选。对于EdgeBoxes（EB），我们使用默认的EB设置来生成候选，该设置将IoU系数调整为0.7。在Fast R-CNN框架下，SS在有58.7%的mAP，而EB有58.6%的mAP。使用了RPN的Fast R-CNN取得了有竞争性的结果，在使用*最多*300个候选下有59.9%的mAP。比起使用SS或者EB而言，使用RPN能形成更快的检测系统，因其使用了共享的卷积计算；更少的候选也相应减少了逐区域全连接层的开销（表5）。

**RPN的剥离实验**。为了调查RPN在作为一个候选方法时的行为，我们开展了几个剥离研究。首先，我们展示了在RPN和Fast R-CNN检测网络上共享卷积层的效果。为了实现该操作，我们在4步的训练过程中，停止于在第二步之后。使用分离的网络轻微地将结果减少到58.7%（RPN+ZF，非共享，表2）。我们观察到这是因为第三步检测器调整过的特征被用于微调RPN后，候选质量得到提升。

接下来，我们解开RPN在训练Fast R-CNN检测网络的影响。为了该目的，我们使用2,000个SS候选和ZF net训练了一个Fast R-CNN模型。我们固定这个检测器，并且通过在测试时改变候选区域来评估检测的mAP。在这个剥离实验中，RPN并没有与检测器共享特征。

在测试时将SS替换成300个RPN的候选使得mAP变为56.8%。在mAP上的损失是因为训练/测试候选时的不一致造成的。这结果作为下面比较的基线来使用。

令人觉得惊讶的是，RPN在测试时使用排名前100的候选时仍能产生竞争性的结果（55.1%），这表明排名靠前的RPN候选是精确的。在另一个极端，使用前6,000个RPN候选（没有NMS）有一个可比较的mAP（55.2%），表明NMS并没有损害到检测的mAP，并且可能减少误报。

接下来，我们分别调查RPN的*分类*和*回归*的输出充当的角色，通过在测试时屏蔽其输出实现。当在测试时移除了*分类*层之后（因此没有使用NMS/排名），我们在无得分的区域中随机采样$N$个候选。在$N=1,000$时，mAP几乎没有改变（55.8%），但是当$N=100$时会降低到44.6%。这表明*分类*得分的确考虑了高排名候选的准确性。

另一方面，当*回归*层在测试时移除之后（所以候选边界直接成为锚点边界），mAP掉至52.1%。这表明高质量的候选主要取决于回归之后的边界。尽管锚点边界有着多种面积和宽高比，但它并不影响检测的精度。

我们还评估了更强大的网络对RPN候选质量的影响。我们使用VGG-16来训练RPN，但使用SS+ZF的检测器。mAP从56.8%（使用RPN+ZF）提高到59.2%（使用RPN+VGG）。这是一个充满希望的结果，因为它表明RPN+VGG的候选质量比起RPN+ZF要更好。因为RPN+ZF的候选会跟SS互相竞争（当训练和测试的方式都一致时，两种方法都达到58.7%），我们可能希望RPN+VGG比SS要更好。接下来的实验会证明这假设。

**VGG-16的表现**。表3展示了VGG-16在候选和检测上的结果。使用RPN+VGG，对于*未共享*特征的结果为68.5%，比起SS的基线要稍微高一些。在上面已经展示过，这是因为通过RPN+VGG生成的候选比起SS要更精确。不像SS这种定义好的方法，RPN能够主动地从更好的网络中训练并从中得益。对于*共享*特征的变化，结果为69.9%——比起强SS基线要更好，并且有着几乎无成本的候选。我们进一步在PASCAL VOC 2007和2012的训练验证集中训练RPN和检测网络。mAP为**73.2%**。图5显示了一些PASCAL 2007测试集中的结果。在PASCAL VOC 2012测试集上（表4），我们的方法在训练在VOC 2007的训练验证+测试集和VOC 2012训练验证集上，取得了**70.4%**的mAP。表6和7展示了详细的数值。

在表5中，我们总结了整个检测系统的运行时间，SS花费1-2秒，其时间取决于内容（一般情况为1.5s），并且在2,000个SS候选上，使用了VGG-16的Fast R-CNN花费320毫秒（或者如果使用在全连接层使用SVD的话则为223毫秒[2]）。我们的系统在使用VGG-16时，在候选和检测的总时间开销为**198ms**。通过卷积特征的共享，RPN只单独花费了10ms进行新增层的计算。得益于更少的候选（每张图片300个候选），我们的逐区域计算也变得更少。我们的系统在使用ZF net时有17fps的帧率。

**超参数的敏感性**。在表8我们研究了锚点的设置。默认情况下我们使用三种面积和三种宽高比（在表8的mAP为69.9%）。如果在每个位置只使用一个锚点，mAP大幅下降了3-4个百分点。如果使用三个面积（和1个宽高比）或者三个宽高比（和1种面积），mAP会更高，阐述了使用多个大小的锚点作为回归参考是一种有效的解决方案。使用仅仅三种面积和一个宽高比（69.8%）在这数据集上跟使用了三种面积和三种宽高比的情况下一样好，表明了面积和宽高比不是检测精度的有关维度。但是我们依旧在我们的设计种采用这两个维度来保持我们系统的灵活性。

在表9中我们比较了等式1中不同的$\lambda$。默认我们使用$\lambda=10$来使得等式1的两项在归一化后大致相同权重。表9显示$\lambda$在两个数量级范围内（1到100），我们的结果只是轻微地受到影响（$\sim1$%）。这阐述了预测结果对大范围的$\lambda$是不敏感的。

在表10中我们研究了测试时的候选数量。

**查全率-交并比（Recall-to-IoU）分析**。接下来我们在ground-truth边界上使用不同的IoU比例来计算候选的查全率。值得注意的是，查全率-交并比指标只跟最终的检测精度有*松散*的相关性[19]，[20]，[21]。使用这种指标更合适于*诊断*候选方法，而不是评估它。

在图4中，我们展示了使用300，1,000，和2,000候选的结果。我们跟SS，EB和MCG进行比较，其中$N$个候选是根据这些方法生成的置信度来选择前$N$个候选的。该图显示，当候选数量从2,000减少到300时，RPN方法表现得很好。这解释了为什么RPN在使用只用300个候选时能够有一个好的最终检测mAP。正如先前我们所分析的那样，这属性主要归因到RPN的*分类*项中。当候选更少时，SS，EB和MCG的查全率掉得更快。

**单一检测阶段与二阶段的候选+检测**。OverFeat论文[9]提出一种检测方法，它在卷积特征图上用回归器和分类器进行窗口滑动。它是一种*单一阶段*，*具体类别*的检测管线，同时我们的是一种*二阶段级联*的方法，它包含了类别无关的候选和具体类别的检测。在OverFeat中，逐区域的特征来自一个面积金字塔中单一宽高比的滑动窗口。这些特征同时用于确定物体的位置和分类。在RPN中，特征来自方形的$(3\times3)$的滑动窗口，并且用不同的面积和宽高比预测相对于锚点的候选。即使两种方法都使用了滑动窗口，但是区域候选任务只是Faster R-CNN的第一阶段任务——下游的Fast R-CNN检测器*参与*了候选的修正。在我们级联的第二阶段，逐区域的特征在候选边界中进行自适应的池化[1]，[2]，更好地覆盖了区域的特征。我们相信这些特征可以用于更精细的检测。

为了比较单一阶段和二阶段的系统，我们通过*一阶段*的Fast R-CNN来*模拟*了OverFeat系统（因此也避开了实现细节上的其他差异）。在这系统中，候选是三种面积（128，256，512）和三种宽高比（1:1，1:2，2:1）的密集滑动窗口。Fast R-CNN训练成用于预测指定类别的得分并且从这些滑动窗口中回归边界位置。因为OverFeat系统采用了图片金字塔，所以我们也使用五种不同的大小下，提取到的卷积特征来进行评估。我们使用在[1]，[2]中的默认大小。

表11比较了两阶段系统和两个不同的单一阶段系统。使用ZF模型，一阶段系统有53.9%的mAP。这比二阶段系统少了4.8%（58.7%）。这个实验证明了级联区域候选和物体检测的效果。同样的观察结果在[2]，[39]中有相应的报告，其中将SS区域候选替换成滑动窗口后会导致$\sim6$%的降低。我们也注意到单一阶段系统因其要处理更多的候选，所以造成了系统变慢。

### 4.2 在MS COCO上的实验

我们在微软的COCO物体识别数据集中显示了更多的结果[12]。这个数据集包含了80个物体分类。我们实验采用了8万张训练集的图片和4万张验证集的图片，以及2万张测试-验证集的图片。我们对于每个IoU$\in[0.5:0.05:0.95]$进行mAP的均值评估（COCO的标准指标可以简单地记为mAP@[.5, .95]）和mAP@0.5（PASCAL VOC的指标）。

我们的系统在这个数据集上做出了一些细微的改变。我们在8-GPU的实现代码中训练我们的模型，RPN的有效mini-batch大小变为8个（每个GPU上1个），Fast R-CNN的大小则为16（每个GPU上2个）。RPN步骤和Fast R-CNN步骤都在学习率为0.003下训练了24万次迭代，然后再用0.0003训练8万次迭代。我们修改了学习率（开始为0.003而非0.001），是因为mini-batch的大小发生了变化。对于锚点，我们使用了三种宽高比和四种面积（添加了$64^2$）主要是为了处理这个数据集中的小型物体。另外，我们的Fast R-CNN步骤中，负样本被定义为那些与ground-truth的最大IoU的区间为$[0,0.5)$的样本，而并非用在[1]，[2]上的$[0.1, 0.5)$。我们注意到在SPPnet系统中[1]，在$[0.1,0.5)$里面的负样本被用于网络的微调，而位于$[0,0.5)$的负样本被用于SVM步骤中遍历hard-negative的样本。但是Fast R-CNN系统[2]抛弃了SVM步骤，所以在$[0,0.1)$之间的负样本没有被浏览到。包含了这些$[0,0.1)$的样本后，Fast R-CNN和Faster R-CNN在COCO数据集上的mAP@0.5都得到提高（但是在PASCAL VOC上则效果不显著）。

剩余的实现细节跟PASCAL VOC一致。特别的是，我们保持使用300个候选和单一大小（$s=600$）进行测试。在COCO数据集中的测试时间依旧为每张图片200ms。

在表12中，我们首次报告了使用本论文实现的Fast R-CNN系统的结果。我们在测试-验证集上的Fast R-CNN有39.3%的mAP@0.5基线，比[2]中报告的要高。我们推测这差距的原因主要是负样本的定义和mini-batch大小的改变。我们也注意到mAP@[.5, .95]只是可比较的。

接下来，我们评估了我们的Faster R-CNN系统。使用COCO的训练集进行训练，Faster R-CNN在测试-验证集上有42.1%的mAP@0.5和21.5%的mAP@[.5, .95]。这跟使用了相同协议（表12）的Fast R-CNN对手相比，在mAP@0.5上提高了2.8%，在mAP@[.5, .95]上提高了2.2%。这表明RPN在更高的IoU阈值上极大提高了定位的精度。使用COCO的训练验证集来训练之下，Faster R-CNN在测试-验证集上有42.7%的mAP@0.5和21.9%的mAP@[.5, .95]。图6显示了一些在MS COCO测试-验证集上的结果。

**Faster RCNN在ILSVRC & COCO 2015比赛**。我们已经展示了Faster R-CNN从更好的特征中获益更多，这得益于RPN完全通过神经网络学习区域候选的事实。这观察结果在一些人将其增加到超过100层时也是基本上有效的[18]。只要把VGG-16替换成101层的残差网络（ResNet-101）[18]，Faster R-CNN系统在COCO验证集上，从41.5%/21.2%的mAP（VGG-16）增加到48.4%/27.2%（ResNet-101）。使用其他与Faster R-CNN正交的提升方法，He等人[18]在COCO测试-验证集上得到了单一模型下55.7%/34.9%的结果，并且有59.0%/37.4%的集成结果，它在COCO 2015物体识别竞赛中赢得了第一名。同样的系统[18]也赢得了ILSVRC 2015物体检测比赛的第一名，以绝对的8.5%的成绩超过了第二名。RPN是ILSVRC 2015定位和COCO 2015分段比赛中的第一名参赛作品中的一块积木，它们的细节分别见[18]和[15]。

### 4.3 从MS COCO到PASCAL VOC

大范围的数据对深度神经网络的提高是至关重要的。接下来，我们研究MS COCO数据集是如何帮助到PASCAL VOC上的检测表现的。

在一个简单的基线中，我们直接用COCO的检测模型评估PASCAL VOC数据集上，而*没有在任何PASCAL数据上进行调整*。这个评估可能是因为COCO的分类是PASCAL VOC分类的一个超集。在该实验中，仅在COCO上出现的分类被忽略掉，并且softmax层只进行20类别加上背景的分类。在这种设置下PASCAL VOC 2007测试集的mAP为76.1%（表13）。即使没利用PASCAL VOC的数据，这个结果也比训练在VOC07+12上的结果要更好（73.2%）。

接着，我们在VOC数据集上微调COCO的检测模型。在本次实验中，COCO模型代替了预训练的ImageNet模型（用来初始化网络权重），然后Faster R-CNN系统如第3.2部分所描述那样调整。做完这些使得在PASCAL VOC 2007测试集上的mAP为78.8%。来自COCO数据集中的外部数据提升了5.6%的mAP。表6展示了训练在COCO+VOC上的模型在每个PASCAL VOC 2007上的单独分类都有着最高的AP。这些提升主要来源于更低的背景误报概率（图7）。相似的提升也能在PASCAL VOC 2012测试集中观察到（表13和表7）。我们注意到，在得到现有的结果时，测试速度依旧保持每张图片约200ms。

## 5. 结论

我们已经展示了RPN用于高效和精确的区域候选生成。通过分享下游检测网络的卷积特征，区域候选步骤几乎是没有开销的。我们的方法能够使统一且基于深度学习的物体检测系统以5-17fps的速度运行。学习到的RPN也能提高区域候选的质量，从而提高物体检测的整体精度。

## 致谢
这项工作在S. Ren还是微软研究部门的实习生时就已经完成了。当R. Girshick还在微软研究部门时，大部分的工作都已经完成了。

## 参考

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” in *Proc. 13th Eur. Conf. Comput. Vis.*, 2014 pp. 346-361.  
[2] R. Girshick, “Fast R-CNN,” in *Proc. IEEE Int. Conf. Comput. Vis.*, 2015, pp. 1440-1448.  
[3] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *Proc. Int. Conf. Learn. Representations*, 2015.  
[4] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders, “Selective search for object recognition,” *Int. J. Comput. Vis.*, vol. 104, no. 2, pp. 154-171, Sep. 2013.  
[5]  R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2014, pp. 580–587.  
[6] C. L. Zitnick and P. Dollar, “Edge boxes: Locating object proposals from edges,” in *Proc. 13th Eur. Conf. Comput. Vis.*, 2014, pp. 391–405.  
[7] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognition*, 2015, pp. 3431–3440.  
[8] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, “Object detection with discriminatively trained part-based models,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 32, no. 9, pp. 1627–1645, Sep. 2010  
[9] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun, “Overfeat: Integrated recognition, localization and detection using convolutional networks,” in *Proc. Int. Conf. Learn. Representations*, 2014  
[10]  S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in *Proc. Neural Inf. Process. Syst.*, 2015, pp. 91–99.  
[11]  M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge Results,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, Jun. 2007.  
[12] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in *Proc. Eur. Conf. Comput. Vis.*, 2014, pp. 740–755.  
[13] S. Song and J. Xiao, “Deep sliding shapes for amodal 3d object detection in RGB-D images,” *arXiv:1511.02300*, 2015.  
[14]  J. Zhu, X. Chen, and A. L. Yuille, “DeePM: A deep part-based model for object detection and semantic part localization,” *arXiv:1511.07131*, 2015.  
[15]  J. Dai, K. He, and J. Sun, “Instance-aware semantic segmentation via multi-task network cascades,” *arXiv:1512.04412*, 2015.  
[16] J. Johnson, A. Karpathy, and L. Fei-Fei, “Densecap: Fully convolutional localization networks for dense captioning,” *arXiv:1511.07571*, 2015.  
[17] D. Kislyuk, Y. Liu, D. Liu, E. Tzeng, and Y. Jing, “Human curation and convnets: Powering item-to-item recommendations on pinterest,” *arXiv:1511.04003*, 2015.  
[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” *arXiv:1512.03385*, 2015.  
[19] J. Hosang, R. Benenson, and B. Schiele, “How good are detection proposals, really?” presented at *Proc. Brit. Mach. Vis. Conf.*, Nottingham, England, 2014.  
[20] J. Hosang, R. Benenson, P. Dollar, and B. Schiele, “What makes for effective detection proposals?” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 38, no. 4, pp. 814–830, Apr. 2015.  
[21] N. Chavali, H. Agrawal, A. Mahendru, and D. Batra, “Object-proposal evaluation protocol is ’gameable’,” *arXiv: 1505.05836*, 2015.  
[22] J. Carreira and C. Sminchisescu, “CPMC: Automatic object segmentation using constrained parametric min-cuts,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 34, no. 7, pp. 1312–1328, Jul. 2012.  
[23] P. Arbelaez, J. Pont-Tuset, J. T. Barron, F. Marques, and J. Malik, “Multiscale combinatorial grouping,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2014, pp. 328–335.  
[24] B. Alexe, T. Deselaers, and V. Ferrari, “Measuring the objectness of image windows,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 34, no. 11, pp. 2189–2202, Nov. 2012.  
[25] C. Szegedy, A. Toshev, and D. Erhan, “Deep neural networks for object detection,” in *Proc. Neural Inform. Process. Syst.*, 2013, pp. 2553–2561.  
[26] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, “Scalable object detection using deep neural networks,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2014, pp. 2155–2162.  
[27] C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, “Scalable, high-quality object detection,” *arXiv:1412.1441 (v1)*, 2015.  
[28] P. O. Pinheiro, R. Collobert, and P. Dollar, “Learning to segment object candidates,” in *Proc. Adv. Neural Inform. Process. Syst.*, 2015, pp. 1981–1989.  
[29] J. Dai, K. He, and J. Sun, “Convolutional feature masking for joint object and stuff segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2015, pp. 3992–4000.  
[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun, “Object detection networks on convolutional feature maps,” *arXiv:1504.06066*, 2015.  
[31] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, “Attention-based models for speech recognition,” in *Proc. Adv. Neural Inform. Process. Syst.*, 2015, pp. 577–585.  
[32] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional neural networks,” in *Proc. 13th Eur. Conf. Comput. Vis.*, 2014, pp. 818–833.  
[33] V. Nair and G. E. Hinton, “Rectified linear units improve restricted Boltzmann machines,” in *Proc. 27th Int. Conf. Mach. Learn.*, 2010, pp. 807–814.  
[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, and A. Rabinovich, “Going deeper with convolutions,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognition*, 2015, pp. 1–9.  
[35] Y. LeCun, et al., “Backpropagation applied to handwritten zip code recognition,” *Neural Comput.*, vol. 1, pp. 541–551, 1989.  
[36] O. Russakovsky, et al., “ImageNet Large Scale Visual Recognition Challenge,” *Int. J. Comput. Vis.*, vol. 115, pp. 211–252, 2015.  
[37] A. Krizhevsky, I. Sutskever, and G. Hinton, “Imagenet classification with deep convolutional neural networks,” in *Proc. Neural Inf. Process. Syst.*, 2012, pp. 1097–1105.  
[38] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” in *Proc. 22nd ACM Int. Conf. Multimedia*, 2014, pp. 675–678.  
[39] K. Lenc and A. Vedaldi, “R-CNN minus R,” in *Proc. Brit. Mach. Vis. Conf.*, 2015, pp. 5.1–5.12.  
[40] D. Hoiem, Y. Chodpathumwan, and Q. Dai, “Diagnosing error in object detectors,” in *Proc. 12th Eur. Conf. Comput. Vis.*, 2012, pp. 340–353.
