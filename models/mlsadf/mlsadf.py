from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn, Tensor
from diffsptk import MLSA, ExcitationGeneration

from models.parallel_wavegan import ParallelWaveGANConfig, ParallelWaveGANGenerator
from yaml_to_dataclass import YamlDataClass


@dataclass
class MLSADFConfig(YamlDataClass):
    source_taylor_order: int
    source_cep_order: int
    filter_taylor_order: int
    filter_cep_order: int
    prenet_a: ParallelWaveGANConfig
    prenet_p: ParallelWaveGANConfig
    mode: Literal[
        "multi-stage",
        "single-stage",
        "freq-domain",
    ] = "multi-stage"


CUSTOM_SR_TO_ALPHA = {
    "8000": 0.31,
    "10000": 0.35,
    "12000": 0.37,
    "16000": 0.42,
    "22050": 0.45,
    "24000": 0.47,  # Added
    "32000": 0.50,
    "44100": 0.53,
    "48000": 0.55,
}


class MLSADF(nn.Module):
    """
    Mel-log Spectrum Approximate Digital Filter
    """

    def __init__(
        self,
        config: MLSADFConfig,
        aux_channels: int,
        sample_rate: int,
        apdc_order: int,
        mcep_order: int,
        hop_length: int,
        n_fft: int,
        without_prenet_a: bool = False,
        without_prenet_p: bool = False
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.alpha = CUSTOM_SR_TO_ALPHA[str(sample_rate)]

        self.with_prenet_a = not without_prenet_a
        self.with_prenet_p = not without_prenet_p

        # validate config
        # if config.prenet is not None:
        #     assert (
        #         config.prenet_a is None and config.prenet_p is None
        #     ), "prenet_a and prenet_p should be None if prenet is not None"
        #     config.prenet_a = config.prenet
        #     config.prenet_p = config.prenet
        # else:
        #     assert (
        #         config.prenet_a is not None and config.prenet_p is not None
        #     ), "prenet_a and prenet_p should not be None if prenet is None"

        # H_p(z) and H_a(z) = 1 - H_p(z)
        if config.mode == "multi-stage":
            self.source_mlsa = MLSA(
                filter_order=apdc_order,
                frame_period=hop_length,
                alpha=self.alpha,
                taylor_order=config.source_taylor_order,
                phase="zero",
                cep_order=config.source_cep_order,
                mode=config.mode,
            )
        elif config.mode == "single-stage":
            self.source_mlsa = MLSA(
                filter_order=apdc_order,
                frame_period=hop_length,
                n_fft=n_fft,
                alpha=self.alpha,
                phase="zero",
                mode=config.mode,
            )
        elif config.mode == "freq-domain":
            self.source_mlsa = MLSA(
                filter_order=apdc_order,
                frame_period=hop_length,
                frame_length=n_fft,
                fft_length=n_fft,
                n_fft=n_fft,
                alpha=self.alpha,
                phase="zero",
                mode=config.mode,
            )
        # H(z)
        if config.mode == "multi-stage":
            self.filter_mlsa = MLSA(
                filter_order=mcep_order,
                frame_period=hop_length,
                alpha=self.alpha,
                taylor_order=config.filter_taylor_order,
                phase="zero",
                cep_order=config.filter_cep_order,
                mode=config.mode,
            )
        elif config.mode == "single-stage":
            self.filter_mlsa = MLSA(
                filter_order=mcep_order,
                frame_period=hop_length,
                n_fft=n_fft,
                alpha=self.alpha,
                phase="zero",
                mode=config.mode,
            )
        elif config.mode == "freq-domain":
            self.filter_mlsa = MLSA(
                filter_order=mcep_order,
                frame_period=hop_length,
                frame_length=n_fft,
                fft_length=n_fft,
                n_fft=n_fft,
                alpha=self.alpha,
                mode=config.mode,
            )
        self.pulse_generator = ExcitationGeneration(
            frame_period=hop_length,
            voiced_region="pulse",
            unvoiced_region="zeros"
        )
        
        self.prenet_p = ParallelWaveGANGenerator(
            config.prenet_p,
            aux_channels=aux_channels
        )

        self.prenet_a = ParallelWaveGANGenerator(
            config.prenet_a,
            aux_channels=aux_channels
        )

    def noise_generator(self, pitch: Tensor) -> Tensor:
        B, T = pitch.shape
        noise = (torch.rand(B, T * self.hop_length) - 0.5) * 2  # [-1, 1]
        return noise.to(device=pitch.device)

    def forward(
        self,
        f0: Tensor,
        ap: Tensor,
        sp: Tensor,
        prenet_a_c: Tensor,
        prenet_p_c: Tensor,
        noise_ratio: float = 0.5,
        sp_rate: float = 1.0,
        f0_rate: float = 1.0,
    ) -> Tensor:
        pitch = self.sample_rate / (f0 * f0_rate)

        # generate aperiodic excitation
        noise = self.noise_generator(pitch).to(next(self.parameters()).dtype)
        exc_a = self.source_mlsa(noise, ap)
        exc_a = exc_a.unsqueeze(1)
        if self.with_prenet_a:
            exc_a = self.prenet_a(exc_a, prenet_a_c)

        # generate periodic excitation
        pulse = self.pulse_generator(pitch).to(next(self.parameters()).dtype)
        exc_p = pulse - self.source_mlsa(pulse, ap)
        exc_p = exc_p.unsqueeze(1)
        if self.with_prenet_p:
            exc_p = self.prenet_p(exc_p, prenet_p_c)

        # mix aperiodic and periodic excitation
        noise_ratio *= 2
        exc = noise_ratio * exc_a + (2 - noise_ratio) * exc_p
        exc = exc.squeeze(1)

        # synthesis
        y = self.filter_mlsa(exc, sp * sp_rate)

        return y.unsqueeze(1)
