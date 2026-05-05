# Ultralytics YOLO 馃殌, AGPL-3.0 license
"""
Block modules
"""
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3')

import torch
import torch.nn.functional as F
from ultralytics.nn.modules.transformer import CrossAttentionModule
from .conv import Conv2
from timm.models.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
        
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=8,
            spline_order=5,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):

        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.SiLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 10
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        flag = True

        if flag:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )

            self.fc2 = KANLinear(
                hidden_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )

            self.fc3 = KANLinear(
                hidden_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            
          
           
            # # TODO
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.fc3 = nn.Linear(hidden_features, hidden_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()

        B, C, H, W = x.size()

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        # x = x.reshape(-1, N, C).contiguous()
        # x = self.dwconv_1(x, H, W)
        x = self.fc2(x)
        # x = x.reshape(B, N, C).contiguous()
        # x = self.dwconv_2(x, H, W)
        x = self.fc3(x)
        # x = x.reshape(B, N, C).contiguous()
        # x = self.dwconv_3(x, H, W)
        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        x = x.reshape(B, H, W, int(C)).permute(0, 3, 1, 2)
        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,):

        B, C, H, W = x.size()

        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.norm2(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.drop_path(self.layer(x, H, W))

        return x

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, z, H):
        x = self.fc(x)
        z = self.fc(z)
        # v -> e
        E = self.v2e(z, H.transpose(1, 2).contiguous()) + x
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c2):
        super().__init__()
        c1 = c2
        self.threshold = 0.65
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6()

    def forward(self, input,input1):
        x = input
        z = input1
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        z = z.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        feature1 = z.clone()
        distance = torch.cdist(feature, feature1)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, z, hg).to(x.device).to(x.dtype)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x

class prediction(nn.Module):
    def __init__(self, cin):
        super(prediction, self).__init__()
        
        self.linear1 = Conv2(cin,cin//8,1,1)
        self.linear2 = Conv2(cin//8,1,1,1)
        self.pool0 = nn.MaxPool2d(kernel_size=20, stride=20, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=40, stride=40, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=80, stride=80, padding=0)
        self.act = nn.ReLU6()
        self.act1 = nn.Sigmoid()

    def forward(self,x):
        B, C, H, W = x.shape
        r = x
        if r.shape[2] is 80:
            r = self.pool2(r)
        elif r.shape[3] is 40:
            r = self.pool1(r)
        else:
            r = self.pool0(r)
        r = self.linear1(r)
        r = self.act(r)
        r = self.linear2(r)
        r = torch.mean(r, dim=0,keepdim=True)
        r = self.act1(r)
        return r

class prediction_offset(nn.Module):
    def __init__(self, cin):
        super(prediction_offset, self).__init__()

        self.linear1 = Conv2(cin, cin // 8, 1, 1)
        self.linear2 = Conv2(cin // 8, 1, 1, 1)
        self.pool0 = nn.MaxPool2d(kernel_size=20, stride=20, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=40, stride=40, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=80, stride=80, padding=0)
        self.act = nn.ReLU6()
        self.act1 = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        if r.shape[2] is 80:
            r = self.pool2(r)
        elif r.shape[3] is 40:
            r = self.pool1(r)
        else:
            r = self.pool0(r)
        r = self.linear1(r)
        r = self.act(r)
        r = self.linear2(r)
        r = torch.mean(r, dim=0, keepdim=True)
        r = self.act1(r)
        return r
        
def get_S(W_adj):
    '''
    W_adj: [B, H, N, N] 
    '''
    D = torch.pow(W_adj.sum(3).float(), -0.5)
    D = torch.diag_embed(D)
   
    S = (W_adj @ D).transpose(-1, -2) @ D

    # D = torch.pow(W_adj.sum(3).float(), -0.5).unsqueeze(-1)
    # D = D @ D.transpose(-1, -2)
    # S = W_adj * D
    return S
    
from torch.nn.parameter import Parameter   
class MultiAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.alpha = 0.6
        self.max_iter_rd = 1
        self.max_iter_dr = 1

        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_d = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_rd = nn.Linear(dim*2, dim)
        self.proj_r = nn.Linear(dim*2, dim)
        self.proj_d = nn.Linear(dim*2, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        device = x.device
        B, N, c = x.shape   
        C = c // 2
        xr, xd = torch.split(x, C, dim=2)
      
        # rgb
        qkv_r = self.qkv_r(xr).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]
        attn_r = ((qr @ kr.transpose(-2, -1)) * self.scale).softmax(dim=-1) 
        S_r = get_S(attn_r)
        out_r = (attn_r @ vr).transpose(1, 2).reshape(B, N, C)
    
        # depth
        qkv_d = self.qkv_d(xd).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qd, kd, vd = qkv_d[0], qkv_d[1], qkv_d[2]
        attn_d = ((qd @ kd.transpose(-2, -1)) * self.scale).softmax(dim=-1) 
        S_d = get_S(attn_d) 
        out_d = (attn_d @ vd).transpose(1, 2).reshape(B, N, C)

        # initial A
        I = torch.eye(N).repeat(self.num_heads, 1, 1).repeat(B, 1, 1, 1)
        I = Parameter(I.cuda()) 
        A = I

        Y = torch.add(attn_r, attn_d)#.softmax(dim=-1)  
        
        device = S_r.device  # 以 S_r 的设备为准，或者使用 self.alpha 所在的设备
        A = A.to(device)
        S_d = S_d.to(device)
        Y = Y.to(device)

        # RGB -- D
        for iter in range(self.max_iter_rd):
            A_rd = self.alpha * (S_r @ A @ S_d.transpose(-1,-2)) + (1 - self.alpha) * Y  
        attn_rd = self.attn_drop(A_rd)
        out_rd = (attn_rd @ vd).transpose(1, 2).reshape(B, N, C)

        # D -- RGB
        for iter in range(self.max_iter_dr):
            A_dr = self.alpha * (S_d @ A @ S_r.transpose(-1,-2)) + (1 - self.alpha) * Y 
        attn_dr = self.attn_drop(A_dr)
        out_dr = (attn_dr @ vr).transpose(1, 2).reshape(B, N, C)
        
      
        return out_dr,out_rd,out_r,out_d
        
class TAAF(nn.Module):
    def __init__(self, cin):
        super(TAAF, self).__init__()

        dim_in = cin
        dim_hidden = cin
        topk = 4
        agg_type = 'sage'
        dropout = 0.3
        pool = 'max'
        
        self.multiatten = MultiAttention(cin)
        self._fc1 = KANBlock(cin)
        self.out = nn.Sequential(nn.Linear(dim_hidden, dim_in), nn.ReLU6())

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)

        if pool == "mean":
            self.readout = global_mean_pool
        elif pool == "max":
            self.readout = global_max_pool
        elif pool == "attn":
            att_net = nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(),
                                    nn.Linear(dim_hidden // 2, 1))
            self.readout = GlobalAttention(att_net)
         
        self.p_sx = prediction_offset(cin*2)
        self.p_sy = prediction_offset(cin*2)
        self.p_sx1 = prediction_offset(cin * 2)
        self.p_sy1 = prediction_offset(cin * 2)
        self.p_ow = prediction(cin*2) 
        self.p_oh = prediction(cin*2)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)

        #self.kan = KANBlock(cin*2)

        self.pool4 = nn.MaxPool2d(kernel_size=10, stride=10, padding=0)

        self.actfun = nn.ReLU6()
        self.sa = CrossAttentionModule(cin, cin)
        self.conv1 = Conv2(cin, cin, 1, 1)
        self.conv2 = Conv2(cin, cin // 2, 1, 1)
        self.conv3 = Conv2(cin // 2, 1, 1, 1)
        self.conv4 = Conv2(18, 1, 1, 1)

        self.supergraph = HyperComputeModule(cin)
      

    def dynamicgraph(self, x1, x2, topk, flag):
        B, C, H, W = x1.size()
       
        
   
        x1 = self._fc1(x1)
        x2 = self._fc1(x2)

        x1 = x1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x1 = (x1 + x1.mean(dim=1, keepdim=True)) * 0.5
        x2 = (x2 + x2.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(x1)
        e_t = self.W_head(x2)
        
        if flag:
          attn_logit = (e_h * self.scale) @ (e_t.transpose(-2, -1))  # 1
            
        else:
          e_h1,e_t1,sr,sd = self.multiatten(torch.cat([e_h,e_t],1).reshape(B, H * W, 2*C))
          attn_logit = ((sr+e_t1) * self.scale) @ ((sd+e_h1).transpose(-2, -1))
          
        
        
        #e_h1,e_t1 = self.multiatten(torch.cat([e_h,e_t],1).reshape(B, H * W, 2*C))
        #attn_logit = ((e_h+e_t1) * self.scale) @ (e_t+e_h1).transpose(-2, -1)  # 1
        

        attn_logit = torch.clamp(attn_logit, min=0.25)
        self.topk = topk

        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)

        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1),
                                                                       e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand * eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)

        h = self.norm(h)
        h = self.out(h)
        h = h.permute(0, 2, 1).view(B, C, H, W)

        return h
    
    def forward(self, x):
        x1 = x[0] #rgb
        x2 = x[1]
        x3 = x[2]
    
        
        B,C,H,W = x1.shape
        if x1.shape[3] is not 80 and x1.shape[3] is not 40 and x1.shape[3] is not 20:
            return x1
        
        ident = x1.clone()
        x1 =torch.sigmoid(F.interpolate(self.dynamicgraph(F.interpolate(x1, (11, 11)), F.interpolate(x2, (11, 11)), 25,flag = True), (H, W)))*x1+x1
        
        z = torch.cat([ident,x3],1)
        a, b, px, py  = self.p_ow(z), self.p_oh(z), self.p_sx(z), self.p_sy(z)
        
        if torch.isnan(a).any():
            a = torch.full_like(a, 0.5)
        if torch.isnan(b).any():
            b = torch.full_like(a, 0.5) 
        if torch.isnan(px).any():
            px = torch.full_like(a, 0.5) 
        if torch.isnan(py).any():
            py = torch.full_like(a, 0.5) 

        h = int(a*H)
        w = int(b*W)
        pOx = int(px*5)
        pOy = int(py*5)

        star_h = (H - h) // 2
        star_w = (W - w) // 2
        mini_x1 = ident[:, :, star_h+pOx:star_h + h+pOx, star_w+pOy:star_w + w+pOy]
        
        mini_x1 = F.interpolate(mini_x1, (11, 11))
        x3 = F.interpolate(x3, (11, 11))
        opt = F.interpolate(self.dynamicgraph(mini_x1, x3, 25, flag = False), (h, w))
       
        ident[:, :, star_h+pOx:star_h + h+pOx, star_w+pOy:star_w + w+pOy] = opt
        
        
        return ident+x1

        


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(3, 16, 3, 2)
        self.cv2 = Conv(16, 32, 3, 2)
        self.c2f1 = C2f(32, 32, 3, True)
        self.cv3 = Conv(32, 64, 3, 2)
        self.c2f2 = C2f(64, 64, 6, True)
        self.cv4 = Conv(64, 128, 3, 2)
        self.c2f3 = C2f(128, 128, 6, True)
        self.cv5 = Conv(128, 256, 3, 2)
        self.c2f4 = C2f(256, 256, 3, True)
        self.sppf = SPPF(256, 256, 5)

    def forward(self, x):  # x:3,h,w
        """Forward pass through RTFPN Block."""
        out = []
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.c2f1(x)
        out.append(x)
        x = self.cv3(x)
        x = self.c2f2(x)
        out.append(x)
        x = self.cv4(x)
        x = self.c2f3(x)
        out.append(x)
        x = self.cv5(x)
        x = self.c2f4(x)
        x = self.sppf(x)
        out.append(x)

        return out


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):  # for input 128 channels
    def __init__(self, cin):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=int(cin / 2), oup=int(cin / 2), expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=int(cin / 2), oup=int(cin / 2), expand_ratio=2)
        self.shffleconv = nn.Conv2d(int(cin / 2), int(cin / 2), kernel_size=1,
                                    stride=1, padding=0, bias=True)
        self.shffleconv1 = nn.Conv2d(int(cin / 2), int(cin / 2), kernel_size=1,
                                     stride=1, padding=0, bias=True)
        self.cin = int(cin / 2)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, x1, x2):  # in_c = cin/2
        temp = x1
        x1 = x2
        x2 = temp
        y1 = x1 + self.theta_phi(x2)
        y2 = x2 + self.theta_phi(y1)
        return y1, y2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, cin, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(cin) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
