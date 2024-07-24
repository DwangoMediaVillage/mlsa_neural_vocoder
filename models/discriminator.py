from torch import Tensor, nn

from config import LossConfig
from models.hifigan.mpd import DiscriminatorP
from models.hifigan.msd import DiscriminatorS
from models.hifigan.utils import discriminator_loss
from models.univnet.mrd import DiscriminatorR


class MultiPeriodAndResolutionDiscriminator(nn.Module):
    def __init__(self, periods: list[int], loss_config: LossConfig):
        super(MultiPeriodAndResolutionDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorP(period) for period in periods
        ] + [
            DiscriminatorR(1024, 120, 600),
            DiscriminatorR(2048, 240, 1200),
            DiscriminatorR(512, 50, 240),
        ])
        self.disc_loss = loss_config.disc_loss if loss_config.disc_loss is not None else 1.0

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

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


class MultiPeriodAndScaleDiscriminator(nn.Module):
    def __init__(self, periods: list[int], loss_config: LossConfig):
        super(MultiPeriodAndScaleDiscriminator, self).__init__()
        self.discriminators_p = nn.ModuleList(
            [DiscriminatorP(period) for period in periods]
        )
        self.discriminators_s = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
        self.disc_loss = loss_config.disc_loss if loss_config.disc_loss is not None else 1.0

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators_p):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.discriminators_s):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

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
