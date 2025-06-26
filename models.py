import torch
import torch.nn as nn
import monai
from typing import Sequence, Optional, Tuple, Union, Any
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as tv

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from losses import Identity   # already defined

__all__ = [
    "EfficientNet",
    "AutoEncoder_New",
    "M3T",
]


class EfficientNet(nn.Module):
    """
    3-D EfficientNet-B1 backbone with four task heads:
    OS, PFS, Age regression, and 9-label classification.
    """
    def __init__(self):
        super().__init__()

        self.backbone = monai.networks.nets.EfficientNetBN(
            "efficientnet-b1",
            spatial_dims=3,
            in_channels=3,
            num_classes=1  # will be replaced by Identity
        )
        # Remove original classifier
        self.backbone._fc = Identity()

        self.fc_os   = nn.Linear(1280, 1)
        self.fc_pfs  = nn.Linear(1280, 1)
        self.fc_age  = nn.Linear(1280, 1)
        self.fc_lbls = nn.Linear(1280, 9)

    def forward(self, x):
        feat = self.backbone(x)
        os_out   = self.fc_os(feat)               # OS risk
        pfs_out  = self.fc_pfs(feat)              # PFS risk
        age_out  = torch.sigmoid(self.fc_age(feat))
        label_out = self.fc_lbls(feat)            # 9-label logits
        return os_out, pfs_out, age_out, label_out


class AutoEncoder_New(nn.Module):
    """
    3-D convolutional auto-encoder with survival heads.
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))
        if len(channels) != len(strides):
            raise ValueError("`channels` and `strides` lengths must match")

        # ------------------------------------------------------------------
        # Encoder, intermediate blocks, decoder
        # ------------------------------------------------------------------
        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(
            self.encoded_channels, channels, strides
        )
        self.intermediate, self.encoded_channels = self._get_intermediate_module(
            self.encoded_channels, num_inter_units
        )
        self.decode, _ = self._get_decode_module(
            self.encoded_channels, decode_channel_list, strides[::-1] or [1]
        )

        # ------------------------------------------------------------------
        # Survival-specific fully-connected network
        # ------------------------------------------------------------------
        self.survivalnet = self._get_survival_module(self.encoded_channels)
        self.fc1 = nn.Linear(1024, 1)  # OS
        self.fc2 = nn.Linear(1024, 1)  # PFS
        self.fc3 = nn.Linear(1024, 1)  # Age
        self.fc4 = nn.Linear(1024, 9)  # 9-label classifier
        self.bn = nn.Sequential(nn.AvgPool3d(8), nn.Flatten())

    # ======= private helpers ======= #

    def _get_survival_module(self, in_channels: int):
        net = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(256, 512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm3d(1024),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm3d(1024),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Flatten(),
        )
        return net

    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            encode.add_module(
                f"encode_{i}",
                self._get_encode_layer(layer_channels, c, s, False),
            )
            layer_channels = c
        return encode, layer_channels

    def _get_intermediate_module(
        self, in_channels: int, num_inter_units: int
    ) -> Tuple[nn.Module, int]:
        if not self.inter_channels:
            return nn.Identity(), in_channels
        inter = nn.Sequential()
        layer_channels = in_channels
        for i, (dc, dil) in enumerate(zip(self.inter_channels, self.inter_dilations)):
            if self.num_inter_units > 0:
                unit = ResidualUnit(
                    spatial_dims=self.dimensions,
                    in_channels=layer_channels,
                    out_channels=dc,
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=self.num_inter_units,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    dilation=dil,
                    bias=self.bias,
                )
            else:
                unit = Convolution(
                    spatial_dims=self.dimensions,
                    in_channels=layer_channels,
                    out_channels=dc,
                    strides=1,
                    kernel_size=self.kernel_size,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    dilation=dil,
                    bias=self.bias,
                )
            inter.add_module(f"inter_{i}", unit)
            layer_channels = dc
        return inter, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        decode = nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            decode.add_module(
                f"decode_{i}",
                self._get_decode_layer(layer_channels, c, s, i == len(strides) - 1),
            )
            layer_channels = c
        return decode, layer_channels

    def _get_encode_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
        return Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

    def _get_decode_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> nn.Sequential:
        decode = nn.Sequential()
        decode.add_module(
            "conv",
            Convolution(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=is_last and self.num_res_units == 0,
                is_transposed=True,
            ),
        )
        if self.num_res_units > 0:
            decode.add_module(
                "resunit",
                ResidualUnit(
                    spatial_dims=self.dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=1,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    bias=self.bias,
                    last_conv_only=is_last,
                ),
            )
        return decode

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Any:
        x = self.encode(x)
        encoded = self.intermediate(x)
        recon = self.decode(encoded)
        surv_feat = self.survivalnet(encoded)

        os_out = self.fc1(surv_feat)
        pfs_out = self.fc2(surv_feat)
        age_out = torch.sigmoid(self.fc3(surv_feat))
        label_out = self.fc4(surv_feat)

        bottleneck = self.bn(encoded)
        return os_out, pfs_out, age_out, label_out, recon, bottleneck
        

class CNN3DBlock(nn.Module):
    """Two 5×5×5 conv-BN-ReLU layers."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiPlaneSliceProject(nn.Module):
    """
    2-D ResNeXt50 (global-avg-pool) used as projection head on
    3×128 feature slices → token embedding (768 dim).
    """
    def __init__(self, c3d: int, proj_dim: int = 768):
        super().__init__()
        self.gap = tv.resnext50_32x4d(weights="IMAGENET1K_V1").avgpool
        self.proj = nn.Sequential(
            nn.Linear(c3d, 1536),
            nn.ReLU(inplace=True),
            nn.Linear(1536, proj_dim),
        )

    def forward(self, x):
        # split into N=128 slices for each plane
        cor = torch.cat(torch.split(x, 1, dim=2), dim=2)         # C×N×W×H
        sag = torch.cat(torch.split(x, 1, dim=3), dim=3)
        axi = torch.cat(torch.split(x, 1, dim=4), dim=4)
        sc  = (cor * x).permute(0, 2, 1, 3, 4)                  # N×C×W×H
        ss  = (sag * x).permute(0, 3, 1, 2, 4)
        sa  = (axi * x).permute(0, 4, 1, 2, 3)
        S   = torch.cat((sc, ss, sa), dim=1)                    # 3N×C×L×L
        feat = self.gap(S).squeeze(-1).squeeze(-1)              # 3N×C
        tokens = self.proj(feat)                                # 3N×d
        return tokens


class EmbeddingLayer(nn.Module):
    """Add cls/sep tokens, plane embeddings and positional encodings."""
    def __init__(self, proj_dim: int = 768, slices: int = 128):
        super().__init__()
        tok = slices * 3
        self.cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.sep = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.cor_emb = nn.Parameter(torch.randn(1, proj_dim))
        self.sag_emb = nn.Parameter(torch.randn(1, proj_dim))
        self.ax_emb  = nn.Parameter(torch.randn(1, proj_dim))
        self.pos_emb = nn.Parameter(torch.randn(tok + 4, proj_dim))

    def forward(self, x):
        # x: B × (3N) × d   (N=128)
        b = x.size(0)
        cls = repeat(self.cls, '1 1 d -> b 1 d', b=b)
        sep = repeat(self.sep, '1 1 d -> b 1 d', b=b)
        # segment layout = CLS | cor 128 | SEP | sag 128 | SEP | ax 128 | SEP
        out = torch.cat(
            (cls,
             x[:, :128], sep,
             x[:, 128:256], sep,
             x[:, 256:], sep),
            dim=1
        )
        # add plane embeddings
        out[:, :130]    += self.cor_emb
        out[:, 130:259] += self.sag_emb
        out[:, 259:]    += self.ax_emb
        out += self.pos_emb
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 8, drop: float = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # 3 × B × h × N × d
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        att = self.drop(att)
        x = (att @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Sequential):
    def __init__(self, dim: int = 768, hidden: int = 768 * 4, drop: float = 0.):
        super().__init__(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 8, drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadAttention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, dim * 4, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, dim: int = 768, heads: int = 8, drop: float = 0.):
        super().__init__(*[
            TransformerBlock(dim, heads, drop) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, dim: int = 768, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.linear(x[:, 0])   # use CLS token


class M3T(nn.Module):
    """
    Full 3-D CNN → multi-plane slice projection → ViT encoder → classification.
    Trunk outputs a 768-D embedding used by multitask heads.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        proj_dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        drop_p: float = 0.15,
    ):
        super().__init__()
        self.cnn3d = CNN3DBlock(in_channels, out_channels)
        self.slice_proj = MultiPlaneSliceProject(out_channels, proj_dim)
        self.embed = EmbeddingLayer(proj_dim)
        self.encoder = TransformerEncoder(depth, proj_dim, heads, drop_p)
        self.cls_head = ClassificationHead(proj_dim, 2)  # not used, kept for completeness

        # multitask heads
        self.fc_os   = nn.Linear(proj_dim, 1)
        self.fc_pfs  = nn.Linear(proj_dim, 1)
        self.fc_age  = nn.Linear(proj_dim, 1)
        self.fc_lbls = nn.Linear(proj_dim, 9)

    def forward(self, x):
        feat3d = self.cnn3d(x)                    # B×C×L×W×H
        tokens = self.slice_proj(feat3d)          # B×(3N)×d
        x_tokens = self.embed(tokens)             # B×(3N+4)×d
        encoded = self.encoder(x_tokens)          # B×(3N+4)×d
        emb = encoded[:, 0]                       # CLS token
        os_out   = self.fc_os(emb)
        pfs_out  = self.fc_pfs(emb)
        age_out  = torch.sigmoid(self.fc_age(emb))
        lbl_out  = self.fc_lbls(emb)
        return os_out, pfs_out, age_out, lbl_out