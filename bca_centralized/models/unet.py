import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False),
            nn.GroupNorm(8, inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False),
            nn.GroupNorm(8, inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi


class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_classes=2, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(ResidualBlock(ch, f))
            ch = f

        self.bottleneck = nn.Sequential(
            ResidualBlock(features[-1], features[-1] * 2),
            nn.Dropout2d(0.3)
        )

        self.ups = nn.ModuleList()
        self.att = nn.ModuleList()
        self.decs = nn.ModuleList()

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.att.append(AttentionGate(f, f, f // 2))
            self.decs.append(ResidualBlock(f * 2, f))

        self.seg_head = nn.Conv2d(features[0], out_channels, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1] * 2, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        skips = []

        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        bottleneck = self.bottleneck(x)

        cls_out = self.cls_head(bottleneck)

        x = bottleneck
        skips = skips[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skips[i]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            skip = self.att[i](x, skip)
            x = torch.cat([skip, x], dim=1)
            x = self.decs[i](x)

        seg_out = self.seg_head(x)

        return seg_out, cls_out


def get_model(device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MultiTaskUNet().to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(device)

    x = torch.randn(2, 1, 160, 160).to(device)
    seg, cls = model(x)

    print(x.shape, seg.shape, cls.shape)
    print(next(model.parameters()).device)
    print(sum(p.numel() for p in model.parameters()))