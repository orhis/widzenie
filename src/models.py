import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 5, pretrained: bool = True):
    """
    Zwraca model torchvision z podmienioną głowicą na num_classes.
    model_name: "efficientnet_b0" | "shufflenet_v2_x1_0"
    """
    model_name = model_name.lower().strip()

    if model_name == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "shufflenet_v2_x1_0":
        if pretrained:
            weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            model = models.shufflenet_v2_x1_0(weights=weights)
        else:
            model = models.shufflenet_v2_x1_0(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("Nieznany model. Użyj: 'efficientnet_b0' albo 'shufflenet_v2_x1_0'.")
