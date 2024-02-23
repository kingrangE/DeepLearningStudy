import torch
import wandb
import argparse

wandb.init(project='wandbPractice')

config = argparse.ArgumentParser()
config.add_argument("--batch_size",default=300)
config.add_argument("")