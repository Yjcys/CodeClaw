# Loss Functions for SRGAN

import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained VGG16 model for feature extraction
vgg16 = models.vgg16(pretrained=True)

# Extract features from layer 5, block 4 (conv4_2)
# This is the same as in the paper's implementation
feature_extractor = nn.Sequential(*list(vgg16.features.children())[:24])


def content_loss(real, generated):
    """Calculate content loss using VGG features"""
    # Extract features from both images
    real_features = feature_extractor(real)
    generated_features = feature_extractor(generated)

    # Calculate Euclidean distance between feature maps
    loss = torch.mean((real_features - generated_features) ** 2)
    return loss


def adversarial_loss(generated, discriminator):
    """Calculate adversarial loss for generator"""
    # Get discriminator predictions
    predictions = discriminator(generated)

    # Calculate log loss
    loss = -torch.log(predictions + 1e-12)
    return loss


def perceptual_loss(real, generated, discriminator):
    """Combine content and adversarial loss for perceptual loss"""
    # Calculate content loss
    content = content_loss(real, generated)

    # Calculate adversarial loss
    adversarial = adversarial_loss(generated, discriminator)

    # Combine losses (paper uses λ=10 for adversarial loss)
    total_loss = content + 10 * adversarial
    return total_loss