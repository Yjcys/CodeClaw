# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from loss import perceptual_loss
from data_loader import get_data_loader
from utils import calculate_psnr, calculate_ssim
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'upscale_factor': 4,
    'data_dir': 'data/train',
    'checkpoint_dir': 'checkpoints',
    'results_dir': 'results',
    'log_dir': 'logs'
}

# Initialize generator and discriminator
generator = Generator(upscale_factor=CONFIG['upscale_factor'])
discriminator = Discriminator()

# Loss functions
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=CONFIG['learning_rate'])
optimizer_d = optim.Adam(discriminator.parameters(), lr=CONFIG['learning_rate'])

# Data loader
train_loader = get_data_loader(CONFIG['data_dir'], batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

# Training loop
def train():
    for epoch in range(CONFIG['num_epochs']):
        for batch in train_loader:
            # Get low-resolution and high-resolution images
            lr_images, hr_images = batch

            # Train discriminator
            optimizer_d.zero_grad()
            
            # Generate high-resolution images
            sr_images = generator(lr_images)
            
            # Calculate adversarial loss for real and generated images
            real_labels = torch.ones(hr_images.size(0), 1)
            fake_labels = torch.zeros(sr_images.size(0), 1)
            
            real_loss = adversarial_criterion(discriminator(hr_images), real_labels)
            fake_loss = adversarial_criterion(discriminator(sr_images.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            
            # Backpropagate
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            
            # Calculate perceptual loss
            g_loss = perceptual_loss(hr_images, sr_images, discriminator)
            
            # Backpropagate
            g_loss.backward()
            optimizer_g.step()
            
            # Log training progress
            logger.info(f'Epoch [{epoch+1}/{CONFIG["num_epochs"]}], Batch [{batch_idx+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

        # Save checkpoint
        torch.save(generator.state_dict(), os.path.join(CONFIG['checkpoint_dir'], f'generator_epoch{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(CONFIG['checkpoint_dir'], f'discriminator_epoch{epoch+1}.pth'))

    logger.info('Training completed')

if __name__ == '__main__':
    train()