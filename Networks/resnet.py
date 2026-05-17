import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SelfAttention(nn.Module):
    """
    Non-local self-attention (SAGAN-style) applied to the spatial feature map
    coming out of layer3.

    Channel dimensions for each variant (input H×W depends on input resolution;
    for a 32×32 CIFAR input the spatial size here is 1×1 — see _forward_impl):

        ResNet-18 / ResNet-34  →  in_dim = 256   (BasicBlock, expansion=1)
        ResNet-50 / ResNet-101 →  in_dim = 1024  (Bottleneck,  expansion=4)
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Compress channels by 8× for Q and K projections (saves compute)
        # ResNet-18/34: in_dim=256  → Q/K channels = 32
        # ResNet-50:    in_dim=1024 → Q/K channels = 128
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # Value keeps the full channel width so the residual add works
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learned scalar that gates how much attention to blend in (starts at 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.temperature = 0.0  # used when mode=False (inference sharpening)
        self.mode = True        # True → learnable gamma; False → fixed temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        # x: (B, C, H, W)
        #   ResNet-18/34: C=256,  H×W depends on input
        #   ResNet-50:    C=1024, H×W same spatial size
        batch_size, C, height, width = x.size()
        N = height * width  # number of spatial positions

        # --- Query projection ---
        # Conv: (B, C, H, W) → (B, C//8, H, W)
        # view: (B, C//8, N)   permute: (B, N, C//8)  — each position is a row vector
        query_out = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)  # (B, N, C//8)

        # --- Key projection ---
        # Conv: (B, C, H, W) → (B, C//8, H, W)  view: (B, C//8, N)
        key_out = self.key_conv(x).view(batch_size, -1, N)  # (B, C//8, N)

        # --- Attention map ---
        # bmm(Q, K): (B, N, C//8) × (B, C//8, N) → (B, N, N)
        # softmax over last dim → each row sums to 1 (attention weights per position)
        attention = self.softmax(torch.bmm(query_out, key_out))  # (B, N, N)

        # --- Value projection ---
        # Conv: (B, C, H, W) → (B, C, H, W)  view: (B, C, N)
        value_out = self.value_conv(x).view(batch_size, -1, N)  # (B, C, N)

        # --- Aggregate ---
        # bmm(V, A^T): (B, C, N) × (B, N, N) → (B, C, N)
        # Each output position = weighted sum of all value vectors
        out = torch.bmm(value_out, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, height, width)             # back to (B, C, H, W)

        # --- Residual blend ---
        if self.mode:
            out = x + out * self.gamma   # gamma starts at 0 so the block is identity at init
        else:
            out = x + out * self.temperature

        return out  # (B, C, H, W) — same shape as input


class BasicBlock(nn.Module):
    """
    The residual building block used by ResNet-18 and ResNet-34.

    Architecture (two 3×3 convolutions):
        x  →  [Conv3x3 → BN → ReLU]  →  [Conv3x3 → BN]  →  (+x)  →  ReLU

    expansion = 1 means the output channels equal `planes` (no channel widening).
    This is in contrast to Bottleneck (expansion=4) used in ResNet-50+, where
    the output channels are planes × 4.

    Dimensionality example (first block of layer2, stride=2):
        inplanes=64, planes=128, stride=2
        Input:  (B, 64,  H,   W)
        conv1:  (B, 128, H/2, W/2)  ← stride=2 spatially downsamples
        conv2:  (B, 128, H/2, W/2)  ← stride=1, no spatial change
        shortcut (downsample): 1×1 conv (B, 64, H, W) → (B, 128, H/2, W/2)
        Output: (B, 128, H/2, W/2)
    """
    expansion: int = 1  # output channels = planes * 1 (no expansion)

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # conv1: (B, inplanes, H, W) → (B, planes, H/stride, W/stride)
        #   stride=1 for all blocks EXCEPT the first block of layers 2-4 (stride=2)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # conv2: (B, planes, H', W') → (B, planes, H', W')  (stride always 1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # downsample: 1×1 conv used to match dimensions for the residual add.
        #   Created by _make_layer when inplanes ≠ planes*expansion OR stride ≠ 1.
        #   Maps (B, inplanes, H, W) → (B, planes, H/stride, W/stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # save input for the residual connection

        # Main path -------------------------------------------------------
        out = self.conv1(x)   # (B, inplanes, H, W) → (B, planes, H/s, W/s)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # (B, planes, H/s, W/s) → (B, planes, H/s, W/s)
        out = self.bn2(out)   # no activation before the residual add
        # -----------------------------------------------------------------

        # Shortcut path: align channel/spatial dims if they changed
        if self.downsample is not None:
            identity = self.downsample(x)  # (B, inplanes, H, W) → (B, planes, H/s, W/s)

        out += identity  # element-wise add; both tensors are now (B, planes, H/s, W/s)
        out = self.relu(out)

        return out  # (B, planes*expansion, H/s, W/s) — for BasicBlock planes*1


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Generic ResNet backbone.

    Tensor shape walkthrough for a 32×32 input (e.g. CIFAR) with batch size B:
    (Standard ImageNet inputs are 224×224; replace the spatial dims accordingly.)

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Layer          │ ResNet-18 / ResNet-34 (BasicBlock)  │  ResNet-50 (Bottleneck) │
    │                │ planes | out_ch | H×W (32px input)  │  out_ch | H×W          │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ Input          │                   B×3×32×32          │  same                  │
    │ conv1 (3×3,s1) │   64  |   64   | B×64×32×32        │  64   | same            │
    │ maxpool (s2)   │        |        | B×64×16×16        │        | same            │
    │ layer1 [×2/×3] │   64  |   64   | B×64×16×16        │  256  | same            │
    │ layer2 [×2/×4] │  128  |  128   | B×128×8×8         │  512  | B×512×8×8       │
    │ layer3 [×2/×6] │  256  |  256   | B×256×4×4         │ 1024  | B×1024×4×4      │
    │ layer4 [×2/×3] │  512  |  512   | B×512×2×2         │ 2048  | B×2048×2×2      │
    │ self_attn      │        |  256   | B×256×4×4         │ 1024  | B×1024×4×4      │
    │ avgpool        │        |        | B×512×1×1         │        | B×2048×1×1      │
    │ flatten        │        |        | B×512             │        | B×2048           │
    │ fc             │        |  C_out | B×num_classes     │        | B×num_classes   │
    └─────────────────────────────────────────────────────────────────────────────┘

    Note: out_ch for BasicBlock = planes * 1; for Bottleneck = planes * 4.
    ResNet-18 layer counts: [2, 2, 2, 2]; ResNet-34: [3, 4, 6, 3].
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_attn: bool = True,
        img_size: int = 32
    ) -> None:
        super(ResNet, self).__init__()
        self.use_attn = use_attn
        self.img_size = img_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # running channel count; updated by _make_layer after each group
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # ── Stem ──────────────────────────────────────────────────────────────
        # Input:  (B, 3,  H,   W)
        # Output: (B, 64, H,   W)   [stride=1, kernel 3×3 — CIFAR variant]
        # Note: standard ImageNet ResNet uses kernel=7, stride=2 here instead.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)  # (B, 64, H, W)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool: (B, 64, H, W) → (B, 64, H/2, W/2)   [kernel=3, stride=2, pad=1]
        # For a 32×32 input:   (B, 64, 32, 32) → (B, 64, 16, 16)
        #self.maxpool = nn.Identity()  # skip pooling for very small inputs
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual stages ───────────────────────────────────────────────────
        # _make_layer updates self.inplanes to planes*block.expansion after each call.
        #
        # layer1: planes=64,  stride=1 (no spatial reduction)
        #   BasicBlock (R18/34): (B, 64, H', W') →  (B,  64, H',  W')   [out=64 *1]
        #   Bottleneck  (R50):   (B, 64, H', W') →  (B, 256, H',  W')   [out=64 *4]
        #   ResNet-18: 2 blocks | ResNet-34: 3 blocks
        self.layer1 = self._make_layer(block, 64, layers[0])

        # layer2: planes=128, stride=2 (halves spatial dims)
        #   BasicBlock: (B,  64, H', W') → (B, 128, H'/2, W'/2)   [out=128*1]
        #   Bottleneck:  (B, 256, H', W') → (B, 512, H'/2, W'/2)   [out=128*4]
        #   ResNet-18: 2 blocks | ResNet-34: 4 blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # layer3: planes=256, stride=2 (halves spatial dims)
        #   BasicBlock: (B, 128, H'', W'') → (B, 256, H''/2, W''/2)  [out=256*1]
        #   Bottleneck:  (B, 512, H'', W'') → (B,1024, H''/2, W''/2)  [out=256*4]
        #   ResNet-18: 2 blocks | ResNet-34: 6 blocks
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # layer4: planes=512, stride=2 (halves spatial dims)
        #   BasicBlock: (B, 256, H''', W''') → (B,  512, H'''/2, W'''/2)  [out=512*1]
        #   Bottleneck:  (B,1024, H''', W''') → (B, 2048, H'''/2, W'''/2)  [out=512*4]
        #   ResNet-18: 2 blocks | ResNet-34: 3 blocks
        #   For a 32×32 input after maxpool: H'''=4, so output is (B, 512, 2, 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # ── Self-Attention ────────────────────────────────────────────────────
        # Applied after layer3 and after layer4.
        # layer3 output channels:
        #   ResNet-18/34: 256
        #   ResNet-50/101/152: 256 * Bottleneck.expansion = 1024
        # layer4 output channels:
        #   ResNet-18/34: 512
        #   ResNet-50/101/152: 512 * Bottleneck.expansion = 2048
        if self.use_attn:
            self.self_attn3 = SelfAttention(256 * block.expansion)
            self.self_attn4 = SelfAttention(512 * block.expansion)
        else:
            self.self_attn3 = nn.Identity()
            self.self_attn4 = nn.Identity()

        # ── Head ──────────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 512 * block.expansion

        self.fc = nn.Linear(self.feat_dim, num_classes)
        self.vis = nn.Linear(self.feat_dim, 2)  # auxiliary 2-D visualisation head
        # self.fc = nn.Linear(2, num_classes)
        # self.feat_dim = 2

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """
        Build one stage (group of residual blocks) with `blocks` blocks.

        The first block may spatially downsample (stride=2) and always adjusts
        channel count.  Subsequent blocks keep the same shape.

        Args:
            block:  BasicBlock (R18/34) or Bottleneck (R50+)
            planes: base channel width for this stage (e.g. 64, 128, 256, 512)
            blocks: number of residual blocks in this stage
            stride: spatial stride applied to the FIRST block only
            dilate: replace stride with dilation (atrous convolution)

        Channel accounting (BasicBlock, expansion=1):
            inplanes_in  → planes * 1 = planes
        Channel accounting (Bottleneck, expansion=4):
            inplanes_in  → planes * 4
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # A downsample (projection shortcut) is needed when:
        #   a) stride ≠ 1 → spatial dims change, OR
        #   b) inplanes ≠ planes*expansion → channel count changes
        # It is a 1×1 conv that matches the shortcut to the main branch output.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block: may have stride>1 and/or channel change → uses downsample
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # Update self.inplanes so subsequent blocks use the correct input channels
        # After first block: inplanes = planes * block.expansion
        self.inplanes = planes * block.expansion

        # Remaining blocks: same channels in/out, stride=1, no downsample needed
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def set_attention_temperature(self, temperature: float) -> None:
        """Set fixed attention scaling for both layer3 and layer4 attention blocks."""
        if isinstance(self.self_attn3, SelfAttention):
            self.self_attn3.set_temperature(temperature)
        if isinstance(self.self_attn4, SelfAttention):
            self.self_attn4.set_temperature(temperature)

    def set_attention_mode(self, mode: bool) -> None:
        """Set attention mode for both layer3 and layer4 attention blocks.

        mode=True  -> use learnable gamma.
        mode=False -> use fixed temperature.
        """
        if isinstance(self.self_attn3, SelfAttention):
            self.self_attn3.set_mode(mode)
        if isinstance(self.self_attn4, SelfAttention):
            self.self_attn4.set_mode(mode)

    def _forward_impl(self, x: Tensor, return_feature=False) -> Tensor:
        """
        Full forward pass with tensor shapes annotated for ResNet-18/34 (BasicBlock).
        Example input: (B, 3, 32, 32)  — CIFAR-style 32×32 images.
        For 224×224 ImageNet inputs, scale the spatial dims accordingly.

        ResNet-18 layer counts: [2, 2, 2, 2]
        ResNet-34 layer counts: [3, 4, 6, 3]
        (The shapes below are the same for both; only depth differs.)
        """
        # ── Stem ──────────────────────────────────────────────────────────────
        # Conv1 (3×3, stride=1): (B, 3, 32, 32) → (B, 64, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)   # (B, 64, 32, 32)
        x = self.relu(x)  # (B, 64, 32, 32)

        # MaxPool (3×3, stride=2): (B, 64, 32, 32) → (B, 64, 16, 16)
        x = self.maxpool(x)

        # ── Residual stages ───────────────────────────────────────────────────
        # layer1: 2 BasicBlocks (R18) / 3 BasicBlocks (R34), planes=64, stride=1
        #   (B, 64, 16, 16) → (B, 64, 16, 16)   [no spatial change, same channels]
        x = self.layer1(x)

        # layer2: 2 BasicBlocks (R18) / 4 BasicBlocks (R34), planes=128, stride=2
        #   (B, 64, 16, 16) → (B, 128, 8, 8)    [halved spatial, doubled channels]
        x = self.layer2(x)

        # layer3: 2 BasicBlocks (R18) / 6 BasicBlocks (R34), planes=256, stride=2
        #   (B, 128, 8, 8)  → (B, 256, 4, 4)    [halved spatial, doubled channels]
        x = self.layer3(x)

        # ── Self-Attention after layer3 ───────────────────────────────────────
        # BasicBlock:  (B, 256, 4, 4)
        # Bottleneck:  (B,1024, 4, 4)
        x = self.self_attn3(x)

        # layer4: 2 BasicBlocks (R18) / 3 BasicBlocks (R34), planes=512, stride=2
        #   BasicBlock: (B, 256, 4, 4)  → (B, 512, 2, 2)
        #   Bottleneck: (B,1024, 4, 4)  → (B,2048, 2, 2)
        x = self.layer4(x)

        # ── Self-Attention after layer4 ───────────────────────────────────────
        # BasicBlock:  (B, 512, 2, 2)
        # Bottleneck:  (B,2048, 2, 2)
        x = self.self_attn4(x)
        
        # ── Head ──────────────────────────────────────────────────────────────
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.fc(x)

        if return_feature:
            return x, y  # x: (B, 512) feature vector; y: (B, num_classes) logits
        else:
            return y     # y: (B, num_classes) logits

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        return self._forward_impl(x, return_feature)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    use_attn = kwargs.pop('use_attn', True)
    img_size = kwargs.pop('img_size', 32)
    model = ResNet(block, layers, use_attn=use_attn, img_size=img_size, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # Handle the classifier mismatch for different num_classes
        if 'fc.weight' in state_dict:
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
        # Handle conv1 size mismatch (usually due to 3x3 conv replacing 7x7 conv)
        if 'conv1.weight' in state_dict and state_dict['conv1.weight'].shape != model.conv1.weight.shape:
            state_dict.pop('conv1.weight')
        
        if not use_attn:
            for k in list(state_dict.keys()):
                if 'self_attn3' in k or 'self_attn4' in k or 'self_attn' in k:
                    state_dict.pop(k, None)

        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# net = resnet18(True,num_classes=3)
# dummy_tensor = torch.zeros(1,3,32,32)

# out = net(dummy_tensor)
