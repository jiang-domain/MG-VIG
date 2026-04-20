# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

#新增Top-K 图池化模块
class TopKPool2d(nn.Module):
    """简单的 Top-k 图池化:
       输入: x (B, C, H, W)
       输出: x_k (B, C, k, 1)，保留 score 最大的 k 个节点
    """
    def __init__(self, in_channels):
        super().__init__()
        # 给每个空间位置打一个标量分数
        self.score_layer = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x, k):
        B, C, H, W = x.shape
        N = H * W

        # (B, 1, H, W) → (B, N)
        score = self.score_layer(x).view(B, N)

        # 选出得分最高的 k 个 index
        topk_score, idx = torch.topk(score, k, dim=1)   # idx: (B, k)

        # 把特征展平到 (B, C, N) 方便 gather
        x_flat = x.view(B, C, N)                        # (B, C, N)

        # 扩展 idx 维度以便按通道 gather
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)   # (B, C, k)

        # 按节点维度 gather，得到 (B, C, k)
        x_k = torch.gather(x_flat, 2, idx_expanded)

        # 这里把 k 个节点当作 1D 网格: (B, C, k, 1)
        x_k = x_k.view(B, C, k, 1)

        return x_k, idx


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.stem = Stem(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

#不改原来的 DeepGCN，单独写一个 MultiScaleDeepGCN，接口保持一致：


class MultiScaleDeepGCN(nn.Module):
    def __init__(self, opt, branch_blocks=None):
        super(MultiScaleDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        use_dilation = opt.use_dilation
        dropout = opt.dropout
        num_classes = opt.n_classes

        # ---- stem & pos_embed：和 DeepGCN 完全一致 ----
        self.stem = Stem(out_dim=channels, act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        # ---- 构建 n_blocks 个 ViG block（Grapher + FFN） ----
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]
        max_dilation = 196 // max(num_knn)

        blocks = []
        for i in range(self.n_blocks):
            if use_dilation:
                dilation = min(i // 4 + 1, max_dilation)
            else:
                dilation = 1
            blocks.append(
                Seq(
                    Grapher(
                        channels,
                        num_knn[i],
                        dilation,
                        conv,
                        act,
                        norm,
                        bias,
                        stochastic,
                        epsilon,
                        1,
                        drop_path=dpr[i],
                    ),
                    FFN(
                        channels,
                        channels * 4,
                        act=act,
                        drop_path=dpr[i],
                    ),
                )
            )
        self.backbone = nn.ModuleList(blocks)

        # ---- Top-K 池化：根据 branch_blocks 选择不同的分支 ----
        self.pool = TopKPool2d(in_channels=channels)

        # ---- 三个分支的 gate 参数（G1、G2、G3）----
        #self.gates = nn.Parameter(torch.zeros(3))
        #self.gates = nn.Parameter(torch.tensor([-4.0, -4.0, -4.0]))
        self.gates = nn.Parameter(torch.full((3, channels, 1, 1), -4.0))

        # ---- 分类头：和 DeepGCN 的 prediction 保持一致 ----
        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_classes, 1, bias=True),
        )

        self.model_init()

        # 设置 branch_blocks：默认是 [3, 7, 11]，适用于 vig_s_224_gelu
        # 可以根据不同的需求传入 branch_blocks 参数
        self.branch_blocks = branch_blocks if branch_blocks is not None else [2, 5, 8]

    def model_init(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        # 1) 处理stem和位置编码
        x = self.stem(inputs) + self.pos_embed      # (B, C, 14, 14)

        G1 = G2 = G3 = None

        # 2) 依次通过每一个ViG block
        for i, blk in enumerate(self.backbone):
            x = blk(x)

            # 根据 branch_blocks 指定池化的层位置
            if i == self.branch_blocks[0]:  # 第1个分支
                x1 = x
                x1_k, _ = self.pool(x1, k=128)  # 选取128个节点
                G1 = F.adaptive_avg_pool2d(x1_k, 1)  # (B, C, 1, 1)

            elif i == self.branch_blocks[1]:  # 第2个分支
                x2 = x
                x2_k, _ = self.pool(x2, k=64)   # 选取64个节点
                G2 = F.adaptive_avg_pool2d(x2_k, 1)

            elif i == self.branch_blocks[2]:  # 第3个分支
                x3 = x
                x3_k, _ = self.pool(x3, k=32)   # 选取32个节点
                G3 = F.adaptive_avg_pool2d(x3_k, 1)

        # 3) 主干输出
        G_main = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)

        # 4) 融合分支的输出：使用sigmoid门控
        a = torch.sigmoid(self.gates)  # (3, C, 1, 1)
        fused = G_main
        if G1 is not None:
            fused = fused + a[0] * G1
        if G2 is not None:
            fused = fused + a[1] * G2
        if G3 is not None:
            fused = fused + a[2] * G3

        # 5) 分类头
        out = self.prediction(fused)  # (B, num_classes, 1, 1)
        return out.squeeze(-1).squeeze(-1)



@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 320 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 640 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


#原来vig_*_224_gelu 后面，加一个多尺度版本
@register_model
def multiscale_vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=100, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 12           # 与 vig_ti 一致
            self.n_filters = 192
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = MultiScaleDeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

@register_model
def multiscale_vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn  # 邻居数（默认：9）
            self.conv = 'mr'  # 图卷积层类型 {edge, mr}
            self.act = 'gelu'  # 激活函数
            self.norm = 'batch'  # 归一化方式
            self.bias = True  # 是否使用卷积层的偏置
            self.n_blocks = 16  # 设置 vig_s_224_gelu 总共 16 层 vig block
            self.n_filters = 320  # 设置通道数
            self.n_classes = num_classes  # 输出类别数
            self.dropout = drop_rate  # dropout 比率
            self.use_dilation = True  # 是否使用膨胀的 KNN
            self.epsilon = 0.2  # GCN 的随机性控制
            self.use_stochastic = False  # 是否使用随机图卷积
            self.drop_path = drop_path_rate  # 随机深度丢弃率

    # 初始化超参数
    opt = OptInit(**kwargs)
    # 使用 MultiScaleDeepGCN 类
    #model = MultiScaleDeepGCN(opt)
    model = MultiScaleDeepGCN(opt, branch_blocks=[3, 7, 11])
    # 设置模型的默认配置
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

@register_model
def multiscale_vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn  # 邻居数（默认：9）
            self.conv = 'mr'  # 图卷积层类型 {edge, mr}
            self.act = 'gelu'  # 激活函数
            self.norm = 'batch'  # 归一化方式
            self.bias = True  # 是否使用卷积层的偏置
            self.n_blocks = 16  # 设置 vig_s_224_gelu 总共 16 层 vig block
            self.n_filters = 640  # 设置通道数
            self.n_classes = num_classes  # 输出类别数
            self.dropout = drop_rate  # dropout 比率
            self.use_dilation = True  # 是否使用膨胀的 KNN
            self.epsilon = 0.2  # GCN 的随机性控制
            self.use_stochastic = False  # 是否使用随机图卷积
            self.drop_path = drop_path_rate  # 随机深度丢弃率

    # 初始化超参数
    opt = OptInit(**kwargs)
    # 使用 MultiScaleDeepGCN 类
    #model = MultiScaleDeepGCN(opt)
    model = MultiScaleDeepGCN(opt, branch_blocks=[3, 7, 11])
    # 设置模型的默认配置
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

