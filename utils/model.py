import json
import os
from typing import Literal

import torch
from config import AllConfig
from models.hifigan.mpd import MultiPeriodDiscriminator
from models.predictor import EmbedMLSADF_SVS
from models.discriminator import MultiPeriodAndResolutionDiscriminator, MultiPeriodAndScaleDiscriminator
from models.univnet.mrd import MultiResolutionDiscriminator
from models.hifigan.mpd import MultiPeriodDiscriminator
from models.hifigan.msd import MultiScaleDiscriminator


def get_models(
    config: AllConfig,
    device: Literal["cpu", "cuda"],
    onnx=False,
    without_prenet_a=False,
    without_prenet_p=False,
    train=True,
) -> tuple[
    EmbedMLSADF_SVS,
    MultiPeriodDiscriminator |
    MultiScaleDiscriminator |
    MultiResolutionDiscriminator |
    MultiPeriodAndResolutionDiscriminator |
    MultiPeriodAndScaleDiscriminator |
    None
]:
    with open(
        os.path.join(config.preprocess.path.preprocessed_path, "stats.json")
    ) as f:
        stats = json.load(f)
        pitch_data: list[float] = stats["pitch"]
        pitch_mean, pitch_std = pitch_data[2], pitch_data[3]
        mcep_data: list[float] = stats["mcep"]
        mcep_mean, mcep_std = mcep_data[2], mcep_data[3]
        apdc_data: list[float] = stats["apdc"]
        apdc_mean, apdc_std = apdc_data[2], apdc_data[3]

    generator = EmbedMLSADF_SVS(
        model_config=config.model,
        loss_config=config.loss,
        sampling_rate=config.preprocess.audio.sampling_rate,
        apdc_order=config.preprocess.mcep_and_apdc.apdc_channels,
        mcep_order=config.preprocess.mcep_and_apdc.mcep_channels,
        hop_length=config.preprocess.stft.hop_length,
        n_fft=config.preprocess.stft.filter_length,
        f0_mean=pitch_mean,
        f0_std=pitch_std,
        mcep_mean=mcep_mean,
        mcep_std=mcep_std,
        apdc_mean=apdc_mean,
        apdc_std=apdc_std,
        segment_size=config.train.segment_size,
        log_f0=True,
        onnx=onnx,
        without_prenet_a=without_prenet_a,
        without_prenet_p=without_prenet_p,
    ).to(device)

    discriminator = None
    if train and config.model.discriminator_type is not None:
        if config.model.discriminator_type == "multi_period":
            assert config.model.discriminator_periods is not None
            discriminator = MultiPeriodDiscriminator(
                periods=config.model.discriminator_periods,
                loss_config=config.loss,
            )
        elif config.model.discriminator_type == "multi_resolution":
            discriminator = MultiResolutionDiscriminator(loss_config=config.loss)
        elif config.model.discriminator_type == "multi_scale":
            discriminator = MultiScaleDiscriminator(loss_config=config.loss)
        elif config.model.discriminator_type == "multi_period_and_resolution":
            assert config.model.discriminator_periods is not None
            discriminator = MultiPeriodAndResolutionDiscriminator(
                periods=config.model.discriminator_periods,
                loss_config=config.loss
            )
        elif config.model.discriminator_type == "multi_period_and_scale":
            discriminator = MultiPeriodAndScaleDiscriminator(
                periods=config.model.discriminator_periods, loss_config=config.loss
            )
        else:
            raise ValueError("Invalid discriminator type")
        discriminator = discriminator.to(device)

    return generator, discriminator
