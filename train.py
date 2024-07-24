import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from config import AllConfig
from dataset import SynthDataset, SynthDatasetBatchTorch
from models.predictor import EmbedMLSADF_SVS
from utils.checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from utils.model import get_models
from utils.plot import write_data_plot_to_tensorboard
from utils.schedular import WarmupCosineAnnealingLR
from utils.slice import rand_slice_segments, slice_segments
from utils.tools import to_device

if TYPE_CHECKING:
    from models.discriminator import MultiPeriodAndResolutionDiscriminator, MultiPeriodAndScaleDiscriminator
    from models.univnet.mrd import MultiResolutionDiscriminator
    from models.hifigan.mpd import MultiPeriodDiscriminator
    from models.hifigan.msd import MultiScaleDiscriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_epoch = 0

def slice_mcep_and_apdc_and_f0(
    batch: SynthDatasetBatchTorch,
    hop_length: int,
    segment_size: int,
):
    mceps = batch["mceps"]
    apdcs = batch["apdcs"]
    frame_f0s = batch["frame_f0s"]
    mcep_lens = batch["mcep_lens"]
    wavs = batch["wavs"]

    mcep_slices, ids_slice = rand_slice_segments(
        mceps.transpose(1, 2), mcep_lens, segment_size
    )
    mcep_slices = mcep_slices.transpose(1, 2)
    apdc_slices = slice_segments(
        apdcs.transpose(1, 2), ids_slice, segment_size
    ).transpose(1, 2)

    f0_slices = slice_segments(
        frame_f0s.unsqueeze(1), ids_slice, segment_size
    ).transpose(1, 2)

    slice_wavs = slice_segments(
        wavs.unsqueeze(1),
        ids_slice * hop_length,
        segment_size * hop_length
    )

    return mcep_slices, apdc_slices, f0_slices, slice_wavs, ids_slice



def evaluate(
    generator: EmbedMLSADF_SVS,
    dataloader: DataLoader,
    writer: SummaryWriter,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
):
    assert (epoch is not None and step is None) or (epoch is None and step is not None)

    generator.eval()

    all_loss_dict = None
    count = 0

    with torch.no_grad():
        for _batch in dataloader:
            batch: SynthDatasetBatchTorch = to_device(
                _batch, next(generator.parameters()).device
            )
            mcep_slices, apdc_slices, f0_slices, slice_wavs, _ = slice_mcep_and_apdc_and_f0(
                batch,
                generator.hop_length,
                generator.segment_size,
            )

            pred_wavs = generator.forward(
                mceps=mcep_slices,
                apdcs=apdc_slices,
                frame_f0s=f0_slices,
            )
            loss_dict = generator.loss(
                slice_wavs,
                pred_wavs,
                y_d_hat_g=None,
                fmap_r=None,
                fmap_g=None,
            )

            if all_loss_dict is None:
                all_loss_dict = {}
                for k, v in loss_dict.items():
                    all_loss_dict[k] = v.detach()
            else:
                for k, v in loss_dict.items():
                    all_loss_dict[k] += v.detach()

            if count < 5:
                # 記録用にフルサイズのyを取得
                pred_wavs = generator.infer(
                    mceps=batch["mceps"],
                    apdcs=batch["apdcs"],
                    frame_f0s=batch["frame_f0s"].unsqueeze(-1),
                )

            for i, id in enumerate(batch["ids"]):
                if count >= 5:
                    break

                write_data_plot_to_tensorboard(
                    writer,
                    type="eval",
                    epoch=epoch,
                    step=step,
                    id=id,
                    wav=pred_wavs[i, 0, : batch["mcep_lens"][i] * generator.hop_length],
                    sampling_rate=generator.sampling_rate
                )

                count += 1

    for k, v in all_loss_dict.items():
        writer.add_scalar(
            f"{'epoch' if epoch is not None else 'step'}_{k}",
            v.item() / len(dataloader),
            global_step=epoch if epoch is not None else step,
        )
    generator.train()


def train(
    config: AllConfig,
    generator: EmbedMLSADF_SVS,
    discriminator: Union["MultiScaleDiscriminator", "MultiPeriodDiscriminator", "MultiResolutionDiscriminator", "MultiPeriodAndResolutionDiscriminator", "MultiPeriodAndScaleDiscriminator", None],
    optimizer_gen: optim.Optimizer,
    optimizer_disc: optim.Optimizer | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: SummaryWriter,
    eval_writer: SummaryWriter,
    start_epoch: int,
    device: torch.device = torch.device("cpu"),
):
    global save_epoch

    # set scheduler
    scheduler_gen = WarmupCosineAnnealingLR(
        optimizer_gen,
        config.train.optimizer.multiplier,
        config.train.optimizer.warmup_epoch,
        config.train.epochs,
        eta_min=0.0,
        last_epoch=start_epoch - 2
    )
    scheduler_disc = None
    if discriminator is not None and optimizer_disc is not None:
        scheduler_disc = WarmupCosineAnnealingLR(
            optimizer_disc,
            config.train.optimizer.multiplier,
            config.train.optimizer.warmup_epoch,
            config.train.epochs,
            eta_min=0.0,
            last_epoch=start_epoch - 2
        )

    total_bar = tqdm(
        total=config.train.epochs * len(train_loader), position=0, desc="Steps", dynamic_ncols=True
    )
    total_bar.n = (start_epoch - 1) * len(train_loader)

    model_dir = Path(config.train.log_dir)
    segment_size = config.train.segment_size
    hop_length = config.preprocess.stft.hop_length

    # training
    for epoch in range(start_epoch, config.train.epochs + 1):
        save_epoch = epoch

        epoch_loss_dict = None

        generator.train()
        if discriminator is not None:
            discriminator.train()
        for _batch in tqdm(train_loader, position=1, desc=f"Epoch {epoch}", dynamic_ncols=True):
            batch: SynthDatasetBatchTorch = to_device(_batch, device)
            # with autocast(enabled=config.train.fp16_run):

            mcep_slices, apdc_slices, f0_slices, slice_wavs, _ = slice_mcep_and_apdc_and_f0(
                batch,
                hop_length,
                segment_size,
            )

            pred_wavs = generator.forward(
                mceps=mcep_slices,
                apdcs=apdc_slices,
                frame_f0s=f0_slices,
            )

            disc_loss_dict = {}
            y_d_hat_g = None
            fmap_r = None
            fmap_g = None
            if discriminator is not None and optimizer_disc is not None:
                # Train Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = discriminator.forward(
                    slice_wavs, pred_wavs.detach()
                )

                disc_loss_dict = discriminator.loss(
                    y_d_hat_r,
                    y_d_hat_g,
                )

                optimizer_disc.zero_grad()
                disc_loss_dict["disc_total"].backward()

                optimizer_disc.step()

                # Train Generator

                # for adversarial loss / feature matching loss
                _, y_d_hat_g, fmap_r, fmap_g = discriminator.forward(slice_wavs, pred_wavs)

            gen_loss_dict = generator.loss(
                slice_wavs,
                pred_wavs,
                y_d_hat_g=y_d_hat_g,
                fmap_r=fmap_r,
                fmap_g=fmap_g,
            )

            optimizer_gen.zero_grad()
            gen_loss_dict["total"].backward()

            optimizer_gen.step()

            loss_dict = {**disc_loss_dict, **gen_loss_dict}

            if epoch_loss_dict is None:
                epoch_loss_dict = {}
                for k, v in loss_dict.items():
                    epoch_loss_dict[k] = v.detach()
            else:
                for k, v in loss_dict.items():
                    epoch_loss_dict[k] += v.detach()

            if total_bar.n % config.train.interval.log_step == 0:
                log_str = f"Step {total_bar.n}:"
                for k, v in loss_dict.items():
                    writer.add_scalar(
                        f"step_{k}", v.item(), global_step=total_bar.n
                    )
                    log_str += f" {k} = {v.item():.4f},"
                lr = optimizer_gen.param_groups[0]["lr"]
                writer.add_scalar("step_lr", lr, global_step=total_bar.n)
                tqdm.write(log_str[:-1])

            if total_bar.n % config.train.interval.val_step == 0:
                with torch.no_grad():
                    pred_wavs = generator.infer(
                        mceps=batch["mceps"],
                        apdcs=batch["apdcs"],
                        frame_f0s=batch["frame_f0s"].unsqueeze(-1),
                    )
                write_data_plot_to_tensorboard(
                    writer,
                    type="train",
                    epoch=None,
                    step=total_bar.n,
                    id=batch["ids"][0],
                    wav=pred_wavs[0, 0, : batch["mcep_lens"][0] * generator.hop_length],
                    sampling_rate=generator.sampling_rate
                )

                evaluate(generator, val_loader, eval_writer, step=total_bar.n)

            total_bar.update()

        scheduler_gen.step(epoch)
        if scheduler_disc is not None:
            scheduler_disc.step(epoch)

        for k, v in epoch_loss_dict.items():
            writer.add_scalar(
                f"epoch_{k}", v.item() / len(train_loader), global_step=epoch
            )
        lr = optimizer_gen.param_groups[0]["lr"]
        writer.add_scalar("epoch_lr", lr, global_step=epoch)

        evaluate(generator, val_loader, eval_writer, epoch=epoch)

        # save model
        if epoch % config.train.interval.save_epoch == 0:
            save_checkpoint(
                generator,
                optimizer_gen,
                save_epoch,
                os.path.join(model_dir, "G_{}.pth".format(save_epoch)),
            )
            if discriminator is not None and optimizer_disc is not None:
                save_checkpoint(
                    discriminator,
                    optimizer_disc,
                    save_epoch,
                    os.path.join(model_dir, "D_{}.pth".format(save_epoch)),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = AllConfig.load(args.config)

    # set random seed
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model
    generator, discriminator = get_models(config, device)

    # set optimizer
    optimizer_gen = optim.AdamW(
        generator.parameters(),
        lr=config.train.optimizer.lr,
        betas=config.train.optimizer.betas,
        eps=config.train.optimizer.eps,
        weight_decay=config.train.optimizer.weight_decay,
    )
    optimizer_disc = None
    if discriminator is not None:
        optimizer_disc = optim.AdamW(
            discriminator.parameters(),
            lr=config.train.optimizer.lr,
            betas=config.train.optimizer.betas,
            eps=config.train.optimizer.eps,
            weight_decay=config.train.optimizer.weight_decay,
        )

    # scaler = GradScaler(enabled=config.train.fp16_run)

    # set dataloader
    train_dataset = SynthDataset(
        config.preprocess,
        "train.txt",
        config.train.batch_max_len,
        torch_device=device,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataset = SynthDataset(
        config.preprocess,
        "val.txt",
        config.train.batch_max_len,
        torch_device=device,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )

    # set tensorboard
    writer = SummaryWriter(log_dir=config.train.log_dir)
    eval_writer = SummaryWriter(log_dir=f"{config.train.log_dir}/eval")

    # save config
    with open(os.path.join(config.train.log_dir, "config.yaml"), "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False, allow_unicode=True)

    # load checkpoint
    model_dir = Path(config.train.log_dir)
    try:
        start_epoch = load_checkpoint(
            latest_checkpoint_path(model_dir, "G_*.pth"), generator, optimizer_gen
        ) + 1
    except Exception as e:
        print(e)
        start_epoch = 1
    
    _start_epoch = start_epoch
    if discriminator is not None and optimizer_disc is not None:
        try:
            _start_epoch = load_checkpoint(
                latest_checkpoint_path(model_dir, "D_*.pth"), discriminator, optimizer_disc
            ) + 1
        except Exception as e:
            print(e)
            _start_epoch = 1

    
    assert start_epoch == _start_epoch

    try:
        train(
            config,
            generator,
            discriminator,
            optimizer_gen,
            optimizer_disc,
            train_loader,
            val_loader,
            writer,
            eval_writer,
            start_epoch,
            device,
        )
    except KeyboardInterrupt as e:
        print(e)

        save_checkpoint(
            generator,
            optimizer_gen,
            save_epoch,
            os.path.join(model_dir, "G_{}.pth".format(save_epoch)),
        )
        if discriminator is not None and optimizer_disc is not None:
            save_checkpoint(
                discriminator,
                optimizer_disc,
                save_epoch,
                os.path.join(model_dir, "D_{}.pth".format(save_epoch)),
            )


if __name__ == "__main__":
    main()
