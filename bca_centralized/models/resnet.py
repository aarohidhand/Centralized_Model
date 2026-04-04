import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=2, pretrained=True, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.GroupNorm(8, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def unfreeze_for_finetuning(model, optimizer, finetune_lr=1e-5):
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": params, "lr": finetune_lr})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def prepare_input(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.contiguous()


def get_model(num_classes=2, pretrained=True, device="cuda"):
    return build_resnet50(num_classes, pretrained, device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(device=device)

    x = torch.randn(2, 1, 224, 224).to(device)
    x = prepare_input(x)

    out = model(x)

    print(x.shape, out.shape)
    print(next(model.parameters()).device)
    print(sum(p.numel() for p in model.parameters()))