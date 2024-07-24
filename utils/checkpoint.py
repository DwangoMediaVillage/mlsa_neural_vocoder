import os
import glob
from typing import Optional, Tuple

import torch
from torch import nn, optim, Tensor
from tqdm import tqdm


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    printf=tqdm.write,
) -> int:
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint_dict["epoch"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    model.load_state_dict(checkpoint_dict["model"])
    printf("Loaded checkpoint '{}' ({} epoch)".format(checkpoint_path, epoch))
    return epoch


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_path: str,
    printf=tqdm.write,
):
    printf(
        "Saving model and optimizer state at {} epoch to {}".format(
            epoch, checkpoint_path
        )
    )
    state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    printf("Saved!")


def latest_checkpoint_path(dir_path: str, regex: str):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x
