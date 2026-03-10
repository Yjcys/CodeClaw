# SRGAN - Photo-Realistic Single Image Super-Resolution

## Overview
SRGAN is a Generative Adversarial Network (GAN) based framework for photo-realistic single image super-resolution with 4x upscaling factor. This implementation follows the architecture and training methodology described in the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network".

## Features
- Generator network with residual blocks and sub-pixel convolution
- Discriminator network for adversarial training
- Perceptual loss combining content loss and adversarial loss
- Data loading with downsampling and augmentation
- Evaluation metrics (PSNR, SSIM, MOS)
- Training loop with logging and checkpointing

## Directory Structure
```
├── main.py
├── train.py
├── model.py
├── loss.py
├── data_loader.py
├── utils.py
├── config.yaml
├── requirements.txt
├── README.md
├── experiments/
│   ├── set5/
│   ├── set14/
│   └── bsd100/
├── checkpoints/
├── results/
└── logs/
```

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python main.py --mode train
```

### Evaluation
```bash
python main.py --mode evaluate
```

## Results
Results will be saved in the `results/` directory. Metrics (PSNR, SSIM, MOS) will be logged in `logs/`.

## References
- Paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
- Implementation inspired by: [SRGAN PyTorch Implementation](https://github.com/sangwook79/SRGAN-PyTorch)

## License
MIT License