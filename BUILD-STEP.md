# 从头开始构建DETR网络

来自文章[End-to-End Object Detection with Transformers](https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/154305880_816694605586461_2873294970659239190_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=7kbGRIBAKBgAX-YDqBc&_nc_ht=scontent-hkg4-1.xx&oh=00_AfDTc8IvCPPdbB9EaI5dxC1D6BX6XYXmDruSfxcmlNGttQ&oe=6521EF03)

源码参考[facebookresearch/detr](https://github.com/facebookresearch/detr)

解析参考[Bubbliiiing的CSDN](https://blog.csdn.net/weixin_44791964/article/details/128361674)
## 整体结构解析

![DETR](.github/DETR.png)

上面这幅图是论文里的Fig. 2，比较好的展示了整个DETR的工作原理。原文中说DETR无需手工融入先验知识的结构（如NMS非极大值抑制、Anchor生成），实现端到端的目标检测，且检测结果是一次并行输出的。整个DETR可以分为四个部分，分别是：backbone、encoder、decoder以及prediction heads。

backbone是DETR的主干特征提取网络，输入的图片首先会在主干网络里面进行特征提取，提取到的特征可以被称作特征层，是输入图片的特征集合。在主干部分，我们获取了一个特征层进行下一步网络的构建，这一个特征层我称它为有效特征层。

encoder是Transformer的编码网络-特征加强，在主干部分获得的一个有效特征层会首先在高宽维度进行平铺 [CxHxW -> Cx(HxW)]，成为一个特征序列，然后会在这一部分继续使用Self-Attension进行加强特征提取，获得一个加强后的有效特征层。它属于Transformer的编码网络，编码的下一步是解码。

decoder是Transformer的解码网络-特征查询，在encoder部分获得的一个加强后的有效特征层会在这一部分进行解码，解码需要使用到一个非常重要的可学习模块，即上图呈现的object queries。在decoder部分，我们使用一个可学习的查询向量q对加强后的有效特征层进行查询，获得预测结果。

prediction heads是DETR的分类器与回归器，其实就是对decoder获得的预测结果进行全连接，两次全连接分别代表种类和回归参数。图上画了4个FFN，源码中是2个FFN。

因此，整个DETR网络所作的工作就是 特征提取-特征加强-特征查询-预测结果。

## models

首先由文中所述结构，构建模型

### backbone

主干网络