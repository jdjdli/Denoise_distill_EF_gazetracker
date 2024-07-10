123456
"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] B * 196 * 768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchEmbed_TMP(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=32, stride=32)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]  B  64 768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x



class PatchEmbed_event(nn.Module):
    def __init__(self, in_chans=512, embed_dim=768, kernel_size=5, stride=1, flatten=True, norm_layer=False):
        super().__init__()
        self.pos_embedding = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.in_chans = in_chans
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=1)
        # self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()
        # self.attn = SelfAttn(768, 4, 3072, 0.1, 'relu')

    def forward(self, x):
        # allow different input size
        x = x.type(torch.cuda.FloatTensor)
        m = x.squeeze(dim=1)[:, :3, :]
        xyz = self.pos_embedding(x.squeeze(dim=1)[:, :3, :])
        xyz = F.relu(xyz)
        x = torch.cat([xyz, x.squeeze(dim=1)[:, 3:, :]], dim=1)
        B, N, C = x.shape        # 1 1 19 10000 # 32 4096 32
        H = W = int(np.sqrt(N*C//self.in_chans)) # 16 16
        x = x.reshape(B, self.in_chans, H, W)       #  B 512 16 16
        x = self.proj(x)        # B 768 14 14
        # x = self.proj2(x)        # B 768 16 16

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  B*196*768
        # x = self.attn(src1=x, pos_src1=pos_embed)
        x = self.norm(x)
        return x

class PatchEmbed_event_TMP(nn.Module):
    def __init__(self, in_chans=2048, embed_dim=768, kernel_size=10, stride=2, flatten=True, norm_layer=False):
        super().__init__()
        self.pos_embedding = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.in_chans = in_chans
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=3)
        # self.proj2 = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=1)
        # self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()
        # self.attn = SelfAttn(768, 4, 3072, 0.1, 'relu')

    def forward(self, x):
        # allow different input size
        x = x.type(torch.cuda.FloatTensor)
        m = x.squeeze(dim=1)[:, :3, :]
        xyz = self.pos_embedding(x.squeeze(dim=1)[:, :3, :])
        xyz = F.relu(xyz)
        x = torch.cat([xyz, x.squeeze(dim=1)[:, 3:, :]], dim=1)
        B, N, C = x.shape        # 1 1 19 10000 # 32 4096 32
        H = W = int(np.sqrt(N*C//self.in_chans)) # 8 8
        x = x.reshape(B, self.in_chans, H, W)       #  B 2048 8 8
        x = self.proj(x)        # B 768 8 8
        # x = self.proj2(x)        # B 768 16 16

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  B*64*768
        # x = self.attn(src1=x, pos_src1=pos_embed)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return [x, attn]


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):

        batch_size = x.shape[0]
        num_queries = x.shape[1]
        num_keys = y.shape[1]
        x = self.query(x)
        y = self.key(y)
        # 计算注意力分数
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 计算加权和
        V = self.value(y)
        output = torch.bmm(attn_weights, V)

        return output




class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Mlp_tmp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, hidden_features)
        self.fc5 = nn.Linear(hidden_features, hidden_features)
        self.fc6 = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.norm = nn.LayerNorm(hidden_features)
        # self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.norm(x)

        x = self.fc4(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc5(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.norm(x)

        x = self.fc6(x)
        x = self.act(x)
        x = self.drop(x)


        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        attn = self.attn(self.norm1(x))
        x = x + self.drop_path(attn[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return [x, attn[1]]


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.patch_embed_1 = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.patch_embed_tmp = PatchEmbed_TMP(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # self.patch_embed_tmp_ = PatchEmbed_TMP(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.relation = nn.Parameter(torch.zeros(1, self.embed_dim, num_patches + self.num_tokens))
        self.patch_embed_event = PatchEmbed_event(in_chans=512, embed_dim=768, kernel_size=5, stride=1)
        self.patch_embed_event_1 = PatchEmbed_event(in_chans=512, embed_dim=768, kernel_size=5, stride=1)
        
        
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.patch_embed_event_tmp = PatchEmbed_event_TMP(in_chans=512, embed_dim=768, kernel_size=10, stride=2)
        
        # self.cross_attn = CrossAttention(in_dim=embed_dim, out_dim=embed_dim)

        self.mlp = Mlp_tmp(in_features=1536, out_features=1, act_layer=act_layer, drop=0.)
        self.norm1 = nn.LayerNorm(1536)
        # self.norm2 = nn.LayerNorm(768)


        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        self.blocks = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
         
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Conv2d(self.num_features, 1, kernel_size=6, stride=1, padding=1)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor_1 = nn.Linear(768*14*14, 1536)
        # self.regressor_2 = nn.Linear(3072, 1536)
        self.regressor_3 = nn.Linear(1536, 192)
        # self.regressor_4 = nn.Linear(768, 192)
        self.regressor_5 = nn.Linear(192, 2)


        self.norm_input = nn.LayerNorm(768*14*14)
        self.norm_input_0 = nn.LayerNorm(1536)
        self.norm_input_1 = nn.LayerNorm(192)
        self.act = nn.ReLU()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, y, x_tmp_1, y_tmp_1, x_tmp_2, y_tmp_2, x_tmp_3, y_tmp_3, x_tmp_4, y_tmp_4, x_tmp_5, y_tmp_5):
        # [B, C, H, W] -> [B, num_patches, embed_dim]

        # x = torch.concat((x, y), dim=1)
        xx = self.patch_embed(x)
        yy = self.patch_embed_event(y)

        x = self.patch_embed_1(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)

        y = self.patch_embed_event_1(y)  # [B, 196, 768]
        # if self.dist_token is None:
        #     y = torch.cat((cls_token, y), dim=1)  # [B, 197, 768]
        # else:
        #     y = torch.cat((cls_token, self.dist_token.expand(y.shape[0], -1, -1), y), dim=1)
        # x = torch.concat((x, y), dim=1)
        # # x = x + y
        # x = self.cross_attn(x, y)

        x_tmp_1 = self.patch_embed_1(x_tmp_1)  # [B, 196, 768]
        y_tmp_1 = self.patch_embed_event_1(y_tmp_1)# [B, 196, 768]

        x_tmp_2 = self.patch_embed_1(x_tmp_2) 
        y_tmp_2 = self.patch_embed_event_1(y_tmp_2)

        x_tmp_3 = self.patch_embed_1(x_tmp_3) 
        y_tmp_3 = self.patch_embed_event_1(y_tmp_3)

        x_tmp_4 = self.patch_embed_1(x_tmp_4) 
        y_tmp_4 = self.patch_embed_event_1(y_tmp_4)

        x_tmp_5 = self.patch_embed_1(x_tmp_5) 
        y_tmp_5 = self.patch_embed_event_1(y_tmp_5)

        # print(x_tmp_1.size()) # [B, token_size, em_dim]
        # os.system('pause')

        # x_1 = torch.mean(x, dim=1)
        # y_1 = torch.mean(y, dim=1)

        # x_tmp_1_1 = torch.mean(x_tmp_1, dim=1)
        # y_tmp_1_1 = torch.mean(y_tmp_1, dim=1)

        # x_tmp_2_1 = torch.mean(x_tmp_2, dim=1)
        # y_tmp_2_1 = torch.mean(y_tmp_2, dim=1)

        # x_tmp_3_1 = torch.mean(x_tmp_3, dim=1)
        # y_tmp_3_1 = torch.mean(y_tmp_3, dim=1)

        # x_tmp_4_1 = torch.mean(x_tmp_4, dim=1)
        # y_tmp_4_1 = torch.mean(y_tmp_4, dim=1)

        # print(x_tmp_1_1.size())  #[B, em_dim]
        # os.system('pause')

        # x1 = torch.concat((x_1, x_tmp_1_1), dim=1).unsqueeze(1)
        # x2 = torch.concat((x_1, x_tmp_2_1), dim=1).unsqueeze(1)
        # x3 = torch.concat((x_1, x_tmp_3_1), dim=1).unsqueeze(1)
        # x4 = torch.concat((x_1, x_tmp_4_1), dim=1).unsqueeze(1)  #[B, 1, 2*em_dim]


        x1 = torch.concat((x, x_tmp_1), dim=2).unsqueeze(1)
        x2 = torch.concat((x, x_tmp_2), dim=2).unsqueeze(1)
        x3 = torch.concat((x, x_tmp_3), dim=2).unsqueeze(1)
        x4 = torch.concat((x, x_tmp_4), dim=2).unsqueeze(1)  #[B, 1, token_size, 2*em_dim]
        x5 = torch.concat((x, x_tmp_5), dim=2).unsqueeze(1)

        # x_5 = torch.concat((x1, x2, x3, x4), dim=1) #[B, 4, token_size, 2*em_dim]

        x_5 = torch.concat((x1, x2, x3, x4, x5), dim=1)


        # print(x_5.size())

        # print(x_5.size())
        x_5 = self.norm1(x_5)
        tmp_weight = self.mlp(x_5)

        # print(tmp_weight.size())
        
        # print('mlp:',tmp_weight)

        # tmp_weight = torch.nn.functional.softmax(tmp_weight)
        # tmp_weight = tmp_weight.squeeze(2)
        # tmp_weight = torch.mean(tmp_weight, dim=2).squeeze(2)#[B, 4]
        tmp_weight = tmp_weight.squeeze(3)


        # print(tmp_weight.size())
        # os.system('pause')
        tmp_weight = torch.nn.functional.gumbel_softmax(tmp_weight, tau=2, dim=1)
        tmp_weight = torch.mean(tmp_weight, dim=2)
        tmp_weight1 = copy.deepcopy(tmp_weight.detach())
        # tmp_weight1 = tmp_weight.detach()


        mm = torch.max(tmp_weight, dim=1)[1]
        nn = torch.zeros(tmp_weight.size()).to(x_tmp_1.device)

        for i in range(nn.size()[0]):
            nn[i][mm[i]] = 1




        # print(tmp_weight.size())
        # print('max:',tmp_weight)
        
        # os.system('pause')

        # tmp_list_1 = [x_tmp_1, x_tmp_2, x_tmp_3, x_tmp_4]
        # x_tmp = tmp_weight * torch.stack(tmp_list_1, dim=0)
        # x_tmp = torch.mean(x_tmp, dim=0)

        # tmp_list_2 = [y_tmp_1, y_tmp_2, y_tmp_3, y_tmp_4]
        # y_tmp = tmp_weight * torch.stack(tmp_list_2, dim=0)
        # y_tmp = torch.mean(y_tmp, dim=0)
        x_tmp_1_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        x_tmp_2_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        x_tmp_3_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        x_tmp_4_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        x_tmp_5_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)

        y_tmp_1_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        y_tmp_2_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        y_tmp_3_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        y_tmp_4_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)
        y_tmp_5_2 = torch.zeros(x_tmp_1.size()).to(x_tmp_1.device)

        

        for i in range(len(x_tmp_1)):
            # x_tmp_1_2[i] = tmp_weight1[i][0] * x_tmp_1[i]
            # x_tmp_2_2[i] = tmp_weight1[i][1] * x_tmp_2[i]
            # x_tmp_3_2[i] = tmp_weight1[i][2] * x_tmp_3[i]
            # x_tmp_4_2[i] = tmp_weight1[i][3] * x_tmp_4[i]
            # y_tmp_1_2[i] = tmp_weight1[i][0] * y_tmp_1[i]
            # y_tmp_2_2[i] = tmp_weight1[i][1] * y_tmp_2[i]
            # y_tmp_3_2[i] = tmp_weight1[i][2] * y_tmp_3[i]
            # y_tmp_4_2[i] = tmp_weight1[i][3] * y_tmp_4[i]

            x_tmp_1_2[i] = nn[i][0] * x_tmp_1[i]
            x_tmp_2_2[i] = nn[i][1] * x_tmp_2[i]
            x_tmp_3_2[i] = nn[i][2] * x_tmp_3[i]
            x_tmp_4_2[i] = nn[i][3] * x_tmp_4[i]        
            x_tmp_5_2[i] = nn[i][4] * x_tmp_5[i]

            y_tmp_1_2[i] = nn[i][0] * y_tmp_1[i]
            y_tmp_2_2[i] = nn[i][1] * y_tmp_2[i]
            y_tmp_3_2[i] = nn[i][2] * y_tmp_3[i]
            y_tmp_4_2[i] = nn[i][3] * y_tmp_4[i]
            y_tmp_5_2[i] = nn[i][4] * y_tmp_5[i]
        # print(x_tmp_1_2.size())

        # print(y_tmp_4_2)
        # os.system('pause')

        # print(tmp_weight.size())
        # print(x_6.size())
        # print(y_6.size())

        # x_6 = torch.matmul(tmp_weight, x_6)
        # y_6 = torch.matmul(tmp_weight, y_6)

        # print(x_6.size())
        # print(y_6.size())

        # os.system('pause')

        # print(y_tmp_1.size())        
        # print(x_1.size())
        # print(y_1.size())
        # x1 = self.mlp(x_1)
        # print(x1.size())

        # y1 = self.mlp(y_1)
        # print(y1.size())



        

        if self.dist_token is None:
            xx = torch.cat((cls_token, xx), dim=1)  # [B, 197, 768]
        else:
            xx = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        xx = self.pos_drop(xx + self.pos_embed)


        if self.dist_token is None:
            yy = torch.cat((cls_token, yy), dim=1)  # [B, 197, 768]
        else:
            yy = torch.cat((cls_token, self.dist_token.expand(y.shape[0], -1, -1), y), dim=1)
        x = torch.concat((xx, yy), dim=1)


        # if self.dist_token is None:
        #     x_tmp_1_2 = torch.cat((cls_token, x_tmp_1_2), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x_tmp_1_2 = self.pos_drop(x_tmp_1_2 + self.pos_embed)


        x_tmp_4_2 = x_tmp_4_2 + x_tmp_1_2 + x_tmp_2_2 + x_tmp_3_2 + x_tmp_5_2
        x_tmp_4_2 = x_tmp_4_2 + self.temp_pos_embed



        if self.dist_token is None:
            x_tmp_4_2 = torch.cat((cls_token, x_tmp_4_2), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x_tmp_4_2 = self.pos_drop(x_tmp_4_2 + self.pos_embed)


        # if self.dist_token is None:
        #     y_tmp_1_2 = torch.cat((cls_token, y_tmp_1_2), dim=1)  # [B, 197, 768]
        # else:
        #     y_tmp = torch.cat((cls_token, self.dist_token.expand(y_tmp.shape[0], -1, -1), x), dim=1)


        y_tmp_4_2 = y_tmp_4_2 + y_tmp_1_2 + y_tmp_2_2 + y_tmp_3_2 + y_tmp_5_2
        y_tmp_4_2 = y_tmp_4_2 + self.temp_pos_embed
        
        if self.dist_token is None:
            y_tmp_4_2 = torch.cat((cls_token, y_tmp_4_2), dim=1)  # [B, 64, 768]
        else:
            y_tmp = torch.cat((cls_token, self.dist_token.expand(y_tmp.shape[0], -1, -1), x), dim=1)


        x_tmp = torch.concat((x_tmp_4_2, y_tmp_4_2), dim=1)


        






        # if self.dist_token is None:
        #     y_tmp = torch.cat((cls_token, y_tmp), dim=1)  # [B, 64, 768]
        # else:
        #     y_tmp = torch.cat((cls_token, self.dist_token.expand(y_tmp.shape[0], -1, -1), x), dim=1)
        # x_tmp = torch.concat((x_tmp_1_2, x_tmp_4_2), dim=1) # [B, 197*2, 768]


        # if self.dist_token is None:
        #     x_tmp = torch.cat((cls_token, x_tmp), dim=1)  # [B, 197, 768]
        # else:
        #     x_tmp = torch.cat((cls_token, self.dist_token.expand(x_tmp.shape[0], -1, -1), x), dim=1)
        
        # #######
        # #######
        # x_tmp = self.pos_drop(x_tmp + self.pos_embed) #[B, 197, 768]

        # y_tmp = torch.concat((y_tmp_1_2, y_tmp_4_2), dim=1)
        # if self.dist_token is None:
        #     y_tmp = torch.cat((cls_token, y_tmp), dim=1)  # [B, 64, 768]
        # else:
        #     y_tmp = torch.cat((cls_token, self.dist_token.expand(y_tmp.shape[0], -1, -1), x), dim=1)

        # x_tmp = torch.concat((x_tmp, y_tmp), dim=1)

        #########
        #########
        


        # x_tmp_end = self.patch_embed_tmp(x_tmp_end)  # [B, 64, 768]
        # if self.dist_token is None:
        #     x_tmp_end = torch.cat((cls_token, x_tmp_end), dim=1)  # [B, 197, 768]
        # else:
        #     x_tmp_end = torch.cat((cls_token, self.dist_token.expand(x_tmp_end.shape[0], -1, -1), x), dim=1)
        # x_tmp_end = self.pos_drop(x_tmp_end + self.pos_embed)

        # y_tmp_end = self.patch_embed_event_tmp(y_tmp_end)# [B, 64, 768]
        # if self.dist_token is None:
        #     y_tmp_end = torch.cat((cls_token, y_tmp_end), dim=1)  # [B, 197, 768]
        # else:
        #     y_tmp_end = torch.cat((cls_token, self.dist_token.expand(y_tmp_end.shape[0], -1, -1), x), dim=1)
        # x_tmp_end = torch.concat((x_tmp_end, y_tmp_end), dim=1)

        # x_tmp = self.cross_attn(x_tmp, x_tmp_end)

        # x_tmp = self.cross_attn(x_tmp, y_tmp)
        #
        # x = self.cross_attn(x, x_tmp)
        x = torch.concat((x, x_tmp), dim=1)
        # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        attn = x[1]
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])

        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])

        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])
        x = self.blocks(x[0])

        

        x = self.norm(x[0])

        # x_tmp = torch.concat((x_tmp, y_tmp), dim=1)
        # x_tmp = self.patch_embed(x_tmp)  # [B, 196, 768]
        # y_tmp = self.patch_embed_event(y_tmp)
        # # x_tmp = torch.concat((x_tmp, y_tmp), dim=1)
        # # x_tmp = x_tmp + y_tmp
        # x_tmp = self.cross_attn(x_tmp, y_tmp)
        # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x_tmp.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x_tmp = torch.cat((cls_token, x_tmp), dim=1)  # [B, 197, 768]
        # else:
        #     x_tmp = torch.cat((cls_token, self.dist_token.expand(x_tmp.shape[0], -1, -1), x), dim=1)

        # x_tmp = self.pos_drop(x_tmp + self.pos_embed)
        # x_tmp = self.blocks(x_tmp)
        # x_tmp = self.norm(x_tmp)
        #
        # # nn.init.trunc_normal_(self.relation, std=0.02)
        #
        # x = abs(x - x_tmp)

        # extract cls token
        if self.dist_token is None:
            
            n1 = x[:,1:197]
            n2 = x[:,198:394]
            n3 = x[:,395:591]
            n4 = x[:,592:788]
            # n5 = x[:,789:985]
            # n6 = x[:,986:1182]
            pre_list = [n1,n2,n3,n4]
            n = torch.stack(pre_list, dim=0).mean(dim=0)
            n = n.transpose(1, 2)
            # m = n.size(dim=0)
            x = n.reshape([n.size(dim=0), 768, 14, 14])

            return self.pre_logits(x), tmp_weight, attn
        else:
            return x[:, 0], x[:, 1]
        return x

    def forward(self, x, y, x_tmp_1, y_tmp_1, x_tmp_2, y_tmp_2, x_tmp_3, y_tmp_3, x_tmp_4, y_tmp_4, x_tmp_5, y_tmp_5):
        x, tmp_weight, attn = self.forward_features(x, y, x_tmp_1, y_tmp_1, x_tmp_2, y_tmp_2, x_tmp_3, y_tmp_3, x_tmp_4, y_tmp_4, x_tmp_5, y_tmp_5)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            #cls linear
            
            x_1 = x.flatten(2).transpose(1,2)[:,0]
            # print(x_1.size())
            output = self.head(x).flatten(2)[:,0]


            # 将输出pooling成 [B, 768, 1, 1]
            pooled_output = self.global_avg_pooling(x)
            # 将输出扁平化为 [B, 768]
            pooled_output = pooled_output.view(pooled_output.size(0), -1)
            # 通过回归器来预测 (x, y) 坐标

            # 直接展平
            x_flatten = x.reshape(x.size(0), -1) 

            norm_input = self.norm_input(x_flatten)
            coordinates = self.regressor_1(norm_input)
            # coordinates = self.norm_input(coordinates)
            coordinates = self.act(coordinates)

            # coordinates = self.regressor_2(coordinates_act)
            # coordinates = self.act(coordinates)

            coordinates = self.regressor_3(coordinates)
            # coordinates = self.norm_input_1(coordinates)
            coordinates = self.act(coordinates)

            # coordinates = self.regressor_4(coordinates)
            # coordinates = self.act(coordinates)

            
            coordinates = self.regressor_5(coordinates)


        return [output, x_1, tmp_weight, attn, x, coordinates]


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=1,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
