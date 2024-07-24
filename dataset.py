import os
from typing import List, TypedDict
import numpy as np
import torch
from torch.utils.data import Dataset

from dataclasses import dataclass

from config import PreProcessConfig
from utils.pad import pad_1D, pad_2D


@dataclass
class SynthDatasetItem:
    id: str
    f0: np.ndarray
    mcep: np.ndarray
    apdc: np.ndarray
    wav: np.ndarray


class SynthDatasetBatchNumpy(TypedDict):
    ids: list[str]
    frame_f0s: np.ndarray
    mceps: np.ndarray
    apdcs: np.ndarray
    mcep_lens: np.ndarray
    wavs: np.ndarray


class SynthDatasetBatchTorch(TypedDict):
    ids: list[str]
    frame_f0s: torch.Tensor
    mceps: torch.Tensor
    apdcs: torch.Tensor
    mcep_lens: torch.LongTensor
    wavs: torch.Tensor


class SynthDataset(Dataset):
    def __init__(
        self,
        config: PreProcessConfig,
        filename: str,
        max_len: int,
        convert_torch: bool = True,
        torch_device: str = "cpu",
    ):
        self.config = config
        self.data_path = config.path.preprocessed_path
        self.max_len = max_len
        self.convert_torch = convert_torch
        self.torch_device = torch_device

        self.basename: List[str] = []

        with open(os.path.join(self.data_path, filename), "r", encoding="utf-8") as f:
            for line in f.readlines():
                n = line.strip("\n")
                self.basename.append(n)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        pitch_path = os.path.join(
            self.data_path,
            "pitch",
            "pitch-{}.npy".format(basename),
        )
        pitch = np.load(pitch_path)

        wav_path = os.path.join(
            self.data_path,
            "wav",
            "wav-{}.npy".format(basename),
        )
        wav = np.load(wav_path)

        mcep_path = os.path.join(
            self.data_path,
            "mcep",
            "mcep-{}.npy".format(basename),
        )
        mcep = np.load(mcep_path)

        apdc_path = os.path.join(
            self.data_path,
            "apdc",
            "apdc-{}.npy".format(basename),
        )
        apdc = np.load(apdc_path)

        sample = SynthDatasetItem(
            id=basename,
            f0=pitch,
            mcep=mcep,
            apdc=apdc,
            wav=wav,
        )

        return sample

    def collate_fn(self, batch: List[SynthDatasetItem]):
        ids = [sample.id for sample in batch]
        frame_f0s = [sample.f0 for sample in batch]
        mceps = [sample.mcep for sample in batch]
        apdcs = [sample.apdc for sample in batch]
        wavs = [sample.wav for sample in batch]

        hop_length = self.config.stft.hop_length

        for sample_index in range(len(frame_f0s)):
            f0_item = frame_f0s[sample_index]
            if len(f0_item) > self.max_len:
                i = np.random.random_integers(0, len(f0_item) - self.max_len)
                frame_f0s[sample_index] = f0_item[i: i + self.max_len]
                mceps[sample_index] = mceps[sample_index][i: i + self.max_len]
                apdcs[sample_index] = apdcs[sample_index][i: i + self.max_len]
                wavs[sample_index] = wavs[sample_index][i * hop_length: (i + self.max_len) * hop_length]

        mcep_lens = np.array(
            [mcep.shape[0] for mcep in mceps], dtype=np.int64
        )

        frame_f0s = pad_1D(frame_f0s).astype(np.float32)
        mceps = pad_2D(mceps).astype(np.float32)
        apdcs = pad_2D(apdcs).astype(np.float32)
        wavs = pad_1D(wavs)

        res = {
            "ids": ids,
            "frame_f0s": frame_f0s,
            "mceps": mceps,
            "apdcs": apdcs,
            "mcep_lens": mcep_lens,
            "wavs": wavs,
        }

        if self.convert_torch:
            res = {k: torch.from_numpy(v) if k != "ids" else v for k, v in res.items()}

        return res
