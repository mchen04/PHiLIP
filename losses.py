import torch
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        self.resize = resize
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = F.l1_loss(input_features, target_features)
        return loss
