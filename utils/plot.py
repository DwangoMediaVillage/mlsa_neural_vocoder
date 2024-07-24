from typing import Literal, Optional

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter



def write_data_plot_to_tensorboard(
    writer: SummaryWriter,
    type: Literal["train", "eval"],
    epoch: Optional[int],
    step: Optional[int],
    id: str,
    wav: Tensor,
    sampling_rate: int,
):
    tag_prefix = f"{type}/{'epoch' if epoch is not None else 'step'}/{id}"
    global_step = epoch if epoch is not None else step

    writer.add_audio(
        f"{tag_prefix}_wav",
        wav,
        global_step,
        sampling_rate
    )
