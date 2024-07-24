from torch import Tensor, nn
import torch

from models.hifigan.utils import feature_loss, generator_loss
from models.mlsadf import MLSADF
from models.parallel_wavegan import MultiResolutionSTFTLoss

from config import LossConfig, ModelConfig


class EmbedMLSADF_SVS(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        sampling_rate: int,
        apdc_order: int,
        mcep_order: int,
        hop_length: int,
        n_fft: int,
        f0_mean: float,
        f0_std: float,
        mcep_mean: float,
        mcep_std: float,
        apdc_mean: float,
        apdc_std: float,
        segment_size: int,
        log_f0: bool = True,
        onnx: bool = False,
        without_prenet_a: bool = False,
        without_prenet_p: bool = False,
    ):
        super().__init__()
        self.apdc_order = apdc_order
        self.mcep_order = mcep_order
        self.prenet_cond_channels = model_config.prenet_cond_channels

        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.mcep_mean = mcep_mean
        self.mcep_std = mcep_std
        self.apdc_mean = apdc_mean
        self.apdc_std = apdc_std

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.segment_size = segment_size

        self.log_f0 = log_f0
        self.onnx = onnx

        self.mcep_linear = nn.Linear(
            mcep_order + 1,
            model_config.hidden,
        )
        self.apdc_linear = nn.Linear(
            apdc_order + 1,
            model_config.hidden,
        )
        self.f0_linear = nn.Linear(
            1,
            model_config.hidden,
        )
        self.prenet_a_c_linear = nn.Linear(
            model_config.hidden,
            model_config.prenet_cond_channels,
        )
        self.prenet_p_c_linear = nn.Linear(
            model_config.hidden,
            model_config.prenet_cond_channels,
        )

        self.mlsadf = MLSADF(
            config=model_config.mlsadf,
            aux_channels=model_config.prenet_cond_channels,
            sample_rate=sampling_rate,
            apdc_order=apdc_order,
            mcep_order=mcep_order,
            hop_length=hop_length,
            n_fft=n_fft,
            without_prenet_a=without_prenet_a,
            without_prenet_p=without_prenet_p,
        )

        # loss
        self.multi_stft_loss = MultiResolutionSTFTLoss(loss_config.multi_stft_loss)
        self.fm_loss_scale = loss_config.fm_loss if loss_config.fm_loss is not None else 1.0
        self.adv_loss_scale = loss_config.adv_loss if loss_config.adv_loss is not None else 1.0

        self.infer = self.forward

    def denorm_f0(self, f0: Tensor) -> Tensor:
        return f0 * self.f0_std + self.f0_mean

    def denorm_mcep(self, mcep: Tensor) -> Tensor:
        return mcep * self.mcep_std + self.mcep_mean

    def denorm_apdc(self, apdc: Tensor) -> Tensor:
        return apdc * self.apdc_std + self.apdc_mean

    def forward(
        self,
        mceps: Tensor,
        apdcs: Tensor,
        frame_f0s: Tensor,
        excs_mix_rate: float = 0.5,
        sp_rate: float = 1.0,
        f0_rate: float = 1.0,
    ):
        hs_mceps = self.mcep_linear(mceps)
        hs_apdcs = self.apdc_linear(apdcs)
        hs_f0s = self.f0_linear(frame_f0s)

        hs = hs_mceps + hs_apdcs + hs_f0s

        prenet_a_cs = self.prenet_a_c_linear(hs).transpose(1, 2)
        prenet_p_cs = self.prenet_p_c_linear(hs).transpose(1, 2)

        frame_f0s = self.denorm_f0(frame_f0s)
        if self.log_f0:
            frame_f0s = torch.exp(frame_f0s)

        pred_wavs = self.mlsadf(
            f0=frame_f0s.squeeze(-1),
            ap=self.denorm_apdc(apdcs),
            sp=self.denorm_mcep(mceps),
            prenet_a_c=prenet_a_cs,
            prenet_p_c=prenet_p_cs,
            noise_ratio=excs_mix_rate,
            sp_rate=sp_rate,
            f0_rate=f0_rate,
        )

        return pred_wavs

    # def infer(
    #     self,
    #     phonemes: LongTensor,
    #     phoneme_durations: LongTensor,
    #     frame_f0s: Tensor,
    #     frame_volumes: Tensor,
    #     speakers: Optional[Tensor],
    #     excs_mix_rate: float = 0.5,
    # ):
    #     pred_wavs, _, pred_mceps, pred_apdcs, _ = self.forward(
    #         phonemes=phonemes,
    #         phoneme_durations=phoneme_durations,
    #         frame_f0s=frame_f0s,
    #         frame_volumes=frame_volumes,
    #         mcep_lens=None,
    #         speakers=speakers,
    #         excs_mix_rate=excs_mix_rate,
    #         slice=False
    #     )

    #     return pred_wavs, pred_mceps, pred_apdcs

    def loss(
        self,
        wavs: Tensor,
        pred_wavs: Tensor,
        y_d_hat_g: Tensor | None,
        fmap_r: Tensor | None,
        fmap_g: Tensor | None,
    ) -> dict[str, Tensor]:
        sc_loss, mag_loss = self.multi_stft_loss(x=pred_wavs.float(), y=wavs.float())
        multi_stft_loss = sc_loss + mag_loss

        gen_adv_loss = torch.tensor(0.0)
        fm_loss = torch.tensor(0.0)
        if y_d_hat_g is not None:
            gen_adv_loss, _ = generator_loss(y_d_hat_g)
        if fmap_r is not None and fmap_g is not None:
            fm_loss = feature_loss(fmap_r, fmap_g)

        loss = multi_stft_loss + gen_adv_loss * self.adv_loss_scale + fm_loss * self.fm_loss_scale

        return {
            "total": loss,
            "multi_stft": multi_stft_loss,
            "gen_adv": gen_adv_loss,
            "fm": fm_loss,
        }
