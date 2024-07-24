# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Parallel WaveGAN Modules."""

from dataclasses import dataclass
import logging
import math

import numpy as np
import torch
from torch import nn, Tensor

from models.parallel_wavegan.residual_block import Conv1d1x1, WaveNetResidualBlock
from models.parallel_wavegan.upsample import ConvInUpsampleNetwork

from yaml_to_dataclass import YamlDataClass


@dataclass
class ParallelWaveGANConfig(YamlDataClass):
    kernel_size: int
    layers: int
    stacks: int
    residual_channels: int
    gate_channels: int
    skip_channels: int
    aux_context_window: int
    dropout: float
    bias: bool
    use_weight_norm: bool
    use_causal_conv: bool
    upsample_scales: list


class ParallelWaveGANGenerator(nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(
        self,
        config: ParallelWaveGANConfig,
        in_channels=1,
        out_channels=1,
        aux_channels=80,
    ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = config.aux_context_window
        self.layers = config.layers
        self.stacks = config.stacks
        self.kernel_size = config.kernel_size

        # check the number of layers and stacks
        assert self.layers % self.stacks == 0
        layers_per_stack = self.layers // self.stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, config.residual_channels, bias=True)

        # define conv + upsampling network
        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales=config.upsample_scales,
            use_causal_conv=config.use_causal_conv,
            aux_channels=aux_channels,
            aux_context_window=self.aux_context_window,
        )
        self.upsample_factor = np.prod(config.upsample_scales)

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(self.layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = WaveNetResidualBlock(
                kernel_size=self.kernel_size,
                residual_channels=config.residual_channels,
                gate_channels=config.gate_channels,
                skip_channels=config.skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=config.dropout,
                bias=config.bias,
                use_causal_conv=config.use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(config.skip_channels, config.skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(config.skip_channels, out_channels, bias=True),
            ]
        )

        # apply weight norm
        if config.use_weight_norm:
            self.apply_weight_norm()

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        """Calculate forward propagation.

        Args:
            z (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # perform upsampling
        if c is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == z.size(-1), f"{c.size(-1)} vs {z.size(-1)}"

        # encode to hidden representation
        x = self.first_conv(z)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)
