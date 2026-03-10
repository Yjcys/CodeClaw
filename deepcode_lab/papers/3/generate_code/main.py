#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SRGAN Main Script
"""

import argparse
from train import train
from config import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRGAN - Super-Resolution Generative Adversarial Network')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file for resuming training')
    args = parser.parse_args()

    if args.train:
        train(config, resume_path=args.resume)
    elif args.evaluate:
        # TODO: Implement evaluation logic
        pass
    else:
        parser.print_help()