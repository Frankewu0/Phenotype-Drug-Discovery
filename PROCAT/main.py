
import os
import warnings
import argparse
from time import time

import torch

from procat.model.models import ProCAT
from procat.utils.utils import set_seed, mkdir
from procat.configs import get_cfg_defaults
from procat.utils.dataloader import load_datasets,create_dataloaders
from procat.trainer import Trainer

def build_model_and_optimizer(cfg, device):
    model = ProCAT(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MODEL.LR)
    return model, optimizer

def run(cfg_path: str, model_task: str, test_file: str):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    set_seed(cfg.MODEL.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    cfg.model_task = model_task

    print(f"Running on: {device}\n")

    train_set, val_set, test_set = load_datasets('./datasets/bindingdb', test_file)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, train_set, val_set, test_set)

    model, optimizer = build_model_and_optimizer(cfg, device)
    trainer = Trainer(model, optimizer, device, train_loader, val_loader, test_loader, **cfg)

    if model_task == 'train':
        result = trainer.train()
        with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
            wf.write(str(model))
    else:
        result = trainer.test()

    print(f"Results saved to: {cfg.RESULT.OUTPUT_DIR}")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ProCAT for DTI prediction")
    parser.add_argument('--cfg', default='procat/config/config.yaml', type=str)
    parser.add_argument('--model_task', default='predict', choices=['train', 'predict'])
    parser.add_argument('--test_file', default='./datasets/demo_data.csv', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = time()
    run(args.cfg, args.model_task, args.test_file)
    print(f"Total running time: {round(time() - start, 2)}s")



