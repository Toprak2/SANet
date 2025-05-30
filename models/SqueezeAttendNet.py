import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1D(nn.Module):
    """Squeeze‑and‑Excitation for 1‑D feature maps."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))          # (B,C,1)
        return x * w                      # scale channels


class ResBlock1D(nn.Module):
    """
    Tiny ResNet block: Conv‑BN‑ReLU → Conv‑BN → SE → add‑&‑ReLU.
    If stride != 1 or channel dims change, a 1×1 downsample path is used.
    """

    def __init__(self, in_ch, out_ch, k, stride, padding=None, groups=1, se=True):
        super().__init__()
        if padding is None:
            # same‑size by default
            padding = ((stride - 1) + 1 * (k - 1)) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride,
                               padding=padding, groups=groups, bias=False)

        pad2 = (1 * (k - 1)) // 2

        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, stride=1,
                               padding=pad2, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SE1D(out_ch) if se else nn.Identity()
        self.act = nn.GELU()

        self.down = (nn.Identity() if (in_ch == out_ch and stride == 1)
                     else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1,
                      stride=stride, bias=False),
            nn.BatchNorm1d(out_ch))
        )

    def forward(self, x):
        identity = self.down(x)

        out = self.act(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out = self.se(out)

        out = self.act(out + identity)

        return out


class DownsampledAttentionBlock(nn.Module):
    """
    Temporal & channel reduction decoupled + gated residual.
    """

    def __init__(
        self,
        channels: int,
        stride: int = 4,
        attn_dim: int | None = None,
        num_heads: int = 8,
        max_len=500
    ):
        super().__init__()
        self.stride = stride
        attn_dim = attn_dim or channels // 2

        # 1. temporal downsample(keep channels)
        self.down_time = nn.Conv1d(
            channels, channels, kernel_size=stride, stride=stride, padding=0
        )
        self.bn_down = nn.BatchNorm1d(channels)

        # 2. channel downsample
        self.down_ch = nn.Conv1d(channels, attn_dim, kernel_size=1)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, attn_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. attention
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )
        self.ln_attn = nn.LayerNorm(attn_dim)

        # 4. temporal upsample (learnable)
        self.up_time = nn.ConvTranspose1d(
            attn_dim, attn_dim, kernel_size=stride, stride=stride, padding=0
        )

        self.bn_up = nn.BatchNorm1d(attn_dim)

        # 5. channel upsample
        self.up_ch = nn.Conv1d(attn_dim, channels, kernel_size=1)

        # # 6. gate (scalar)   ——————————————————————————
        # self.alpha = nn.Parameter(torch.ones(1,channels, 1))  # start at 0 ⇒ no effect

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        B, C, T = x.shape

        xt = F.gelu(self.bn_down(self.down_time(x)))        # (B, C, T')
        xs = self.down_ch(xt)                               # (B, C_attn, T')
        xs_seq = xs.permute(0, 2, 1)                        # (B, T', C_attn)

        L = xs_seq.size(1)
        xs_seq = xs_seq + self.pos_embed[:, :L]  # positional embeddings

        attn_out, _ = self.mha(xs_seq, xs_seq, xs_seq)
        xs_out = self.ln_attn(xs_seq + attn_out).permute(0,
                                                         # (B, C_attn, T')
                                                         2, 1)

        # (B, C_attn, ≈T)
        xu = F.gelu(self.bn_up(self.up_time(xs_out)))
        # if xu.size(2) > T:
        #     xu = xu[..., :T]

        # gated fusion: y = x + α · Δ
        return x + self.up_ch(xu)


class AttentivePool(nn.Module):
    """Self‑attentive mean + std pooling (Stat‑SAP)."""

    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1, 1)
        )

    def forward(self, x):                 # x:(B,C,T)
        w = torch.softmax(self.score(x), dim=2)   # (B,1,T)
        mu = (x * w).sum(dim=2)                    # weighted mean
        var = ((x - mu.unsqueeze(-1))**2 * w).sum(dim=2)
        std = torch.sqrt(torch.clamp(var, 1e-9))
        return torch.cat([mu, std], dim=1)          # (B, 2C)


class SqueezeAttendNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, class_count=10):
        super().__init__()

        self.stage1 = ResBlock1D(in_channels, 32,  k=64, stride=4, se=True)
        self.stage2 = ResBlock1D(32,          64,  k=32,  stride=2, se=True)
        self.stage3 = ResBlock1D(64,          128, k=6,  stride=2, se=True)
        self.stage4 = ResBlock1D(128, 256,  k=4,  stride=2, se=True)
        self.dab1 = DownsampledAttentionBlock(256, stride=4, attn_dim=128)
        self.stage5 = ResBlock1D(256, latent_dim, k=4, stride=2, se=True)
        self.dab2 = DownsampledAttentionBlock(
            latent_dim, stride=4, attn_dim=256)

        self.pool = AttentivePool(latent_dim)       # → (B, 2*latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, class_count)
        )

    def forward(self, x):               # x:(B,1,T)   T=64 000
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = self.dab1(x)
        # print(x.shape)
        x = self.stage5(x)
        # print(x.shape)
        x = self.dab2(x)
        # print(x.shape)
        z = self.pool(x)                # (B, 2*latent_dim)
        # print(z.shape)
        return self.classifier(z)
