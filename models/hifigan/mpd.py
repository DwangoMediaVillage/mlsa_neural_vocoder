import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from config import LossConfig
from models.hifigan.utils import discriminator_loss, get_padding

LRELU_SLOPE = 0.1

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int], loss_config: LossConfig):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in periods]
        )
        self.disc_loss = loss_config.disc_loss if loss_config.disc_loss is not None else 1.0

    def forward(self, y, y_hat):
        ret_rs = []
        ret_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            ret_r, fmap_r = disc(y)
            ret_g, fmap_g = disc(y_hat)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            ret_rs.append(ret_r)
            ret_gs.append(ret_g)

        return ret_rs, ret_gs, fmap_rs, fmap_gs

    def loss(
        self,
        y_d_hat_r: Tensor,
        y_d_hat_g: Tensor,
    ) -> dict[str, Tensor]:
        disc_adv_loss, _, _ = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )

        loss = disc_adv_loss * self.disc_loss

        return {
            "disc_total": loss,
            "disc_adv_loss": disc_adv_loss,
        }
