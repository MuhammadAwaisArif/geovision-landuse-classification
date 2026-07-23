import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class CNNTransformerHybrid(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transformer_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.token_projection = nn.Linear(512, transformer_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 50, transformer_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, num_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, inputs):
        features = self.backbone(inputs)
        if features.shape[-2:] != (7, 7):
            raise ValueError("Hybrid expects 224 x 224 inputs producing a 7 x 7 feature map")
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        class_tokens = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded = self.transformer(tokens)
        return self.head(encoded[:, 0])


def create_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "hybrid":
        return CNNTransformerHybrid(num_classes, pretrained=pretrained)
    if name == "vit":
        import timm

        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")

