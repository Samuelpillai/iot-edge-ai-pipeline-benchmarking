# Feature Extraction with MobileNetV2

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class MobileNetVectorizer:
    def __init__(self):
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v2(weights=weights)
        self.model = torch.nn.Sequential(
            *list(mobilenet.features),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_vector(self, image):
        with torch.no_grad():
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dim
            output = self.model(input_tensor)
            return output.view(-1).numpy().tolist()  # 1280-dim vector