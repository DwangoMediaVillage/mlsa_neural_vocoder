from typing import Literal
from models.mlsadf.mlsadf import MLSADFConfig
from models.parallel_wavegan import MultiResolutionSTFTLossConfig
from yaml_to_dataclass import YamlDataClass

from dataclasses import dataclass


@dataclass
class PreProcessPath(YamlDataClass):
    data_path: str
    preprocessed_path: str


@dataclass
class PreProcessAudio(YamlDataClass):
    sampling_rate: int
    max_wav_value: float
    trim_top_db: int


@dataclass
class PreProcessSTFT(YamlDataClass):
    filter_length: int
    hop_length: int
    win_length: int


@dataclass
class PreProcessCepsAndAperiodicity(YamlDataClass):
    mcep_channels: int
    apdc_channels: int


@dataclass
class PreProcessMel(YamlDataClass):
    n_mel_channels: int
    n_mel_channels_loss: int
    mel_fmin: int
    mel_fmax: int
    mel_fmax_loss: int | None


@dataclass
class PreProcessConfig(YamlDataClass):
    path: PreProcessPath
    val_size: int
    audio: PreProcessAudio
    stft: PreProcessSTFT
    mel: PreProcessMel
    mcep_and_apdc: PreProcessCepsAndAperiodicity


@dataclass
class ModelConfig(YamlDataClass):
    hidden: int
    prenet_cond_channels: int
    mlsadf: MLSADFConfig
    discriminator_type: Literal[
        None,
        "multi_period",
        "multi_resolution",
        "multi_scale",
        "multi_period_and_resolution",
        "multi_period_and_scale"
    ] = None
    discriminator_periods: list | None = None


@dataclass
class TrainOptimizerConfig(YamlDataClass):
    lr: float
    betas: tuple[float, float]
    lr_decay: float
    weight_decay: float
    eps: float
    multiplier: float
    warmup_epoch: int


@dataclass
class TrainIntervalConfig(YamlDataClass):
    save_epoch: int
    log_step: int
    val_step: int


@dataclass
class TrainConfig(YamlDataClass):
    seed: int
    optimizer: TrainOptimizerConfig
    batch_size: int
    batch_max_len: int
    segment_size: int
    epochs: int
    interval: TrainIntervalConfig
    log_dir: str
    fp16_run: bool


@dataclass
class LossConfig(YamlDataClass):
    multi_stft_loss: MultiResolutionSTFTLossConfig
    lambda_feat: float
    fm_loss: float | None = None
    adv_loss: float | None = None
    disc_loss: float | None = None


@dataclass
class AllConfig(YamlDataClass):
    preprocess: PreProcessConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
