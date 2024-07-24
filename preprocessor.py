import multiprocessing
import os
from pathlib import Path
import random
import json
import argparse

import librosa
import resampy
import torch
from sklearn.preprocessing import StandardScaler

import numpy as np
import pysptk
import pyworld as pw
import pydub
from pydub.silence import split_on_silence
from tqdm import tqdm

from config import AllConfig, PreProcessConfig
from models.mlsadf import CUSTOM_SR_TO_ALPHA
from utils.mel_processing import spectrogram_torch as spectrogram


from typing import Tuple, List


MAX_WAV_VALUE = 32768.0


def load_wav(
    full_path: str, sampling_rate: int, hop_length: int, split: bool
) -> np.ndarray:
    data, sr = librosa.load(full_path, sr=None, mono=True)
    # resampling by outside of librosa and showing warning
    if sampling_rate != sr:
        # tqdm.write(f"sampling rate is different(required: {sampling_rate}, actually: {sr}, file: {full_path}), auto converted by script.")
        data = resampy.resample(data, sr, sampling_rate, filter="kaiser_best")

    split_data = []

    if split:
        if max(data) <= 1.0:
            data = data * MAX_WAV_VALUE

        sound = pydub.AudioSegment(
            data=data.astype("int16").tobytes(),
            sample_width=2, # 2byte (16bit)
            frame_rate=sampling_rate,
            channels=1,
        )

        chunks = split_on_silence(
            sound, 
            min_silence_len = 500,
            silence_thresh = -45,
            keep_silence = 200
        )

        split_data = []

        for chunk in chunks:
            wav = np.array(chunk.get_array_of_samples()).astype("float32") / MAX_WAV_VALUE
            data_len = hop_length * (len(wav) // hop_length)
            wav = wav[:data_len]
            split_data.append(wav)
    else:
        data_len = hop_length * (len(data) // hop_length)
        split_data = [data[:data_len]]

    return split_data


class Preprocessor:
    def __init__(self, config: PreProcessConfig, split):
        self.config = config
        self.in_dir = config.path.data_path
        self.out_dir = config.path.preprocessed_path
        self.val_size = config.val_size
        self.sampling_rate = config.audio.sampling_rate
        self.alpha = CUSTOM_SR_TO_ALPHA[str(self.sampling_rate)]
        self.max_wav_value = config.audio.max_wav_value
        self.hop_length = config.stft.hop_length
        self.pitch_scaler = StandardScaler()
        self.mcep_scaler = StandardScaler()
        self.apdc_scaler = StandardScaler()
        self.consonant_duration_scaler = StandardScaler()
        self.split = split

    def build_from_path(self) -> List[str]:
        os.makedirs((os.path.join(self.out_dir, "wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mcep")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "apdc")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)

        print("Processing Data ...")
        out: List[str] = []
        n_frames = 0

        wavs = list(
            filter(
                lambda x: x.endswith(".wav"),
                os.listdir(self.in_dir)
            )
        )

        with multiprocessing.Pool() as pool:
            for data in tqdm(
                pool.imap_unordered(self.process_utterance, wavs),
                desc="Files",
                position=1,
                total=len(wavs),
            ):
                for i, pitch in enumerate(data["pitches"]):
                    if pitch.size > 0:
                        self.pitch_scaler.partial_fit(
                            pitch
                        )
                    self.mcep_scaler.partial_fit(
                        data["mceps"][i]
                    )
                    self.apdc_scaler.partial_fit(
                        data["apdcs"][i]
                    )
                    out.append(f"{data['basename']}_{i}")

        print("Computing statistic quantities ...")
        pitch_mean = self.pitch_scaler.mean_[0]
        pitch_std = self.pitch_scaler.scale_[0]
        mcep_mean = self.mcep_scaler.mean_[0]
        mcep_std = self.mcep_scaler.scale_[0]
        apdc_mean = self.apdc_scaler.mean_[0]
        apdc_std = self.apdc_scaler.scale_[0]

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        mcep_min, mcep_max = self.normalize(
            os.path.join(self.out_dir, "mcep"), mcep_mean, mcep_std
        )
        apdc_min, apdc_max = self.normalize(
            os.path.join(self.out_dir, "apdc"), apdc_mean, apdc_std
        )

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "mcep": [
                    float(mcep_min),
                    float(mcep_max),
                    float(mcep_mean),
                    float(mcep_std),
                ],
                "apdc": [
                    float(apdc_min),
                    float(apdc_max),
                    float(apdc_mean),
                    float(apdc_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(
        self, basename: str
    ) -> dict:
        split_data = load_wav(
            os.path.join(self.config.path.data_path, basename),
            self.config.audio.sampling_rate,
            self.config.stft.hop_length,
            self.split
        )

        length = 0
        split_count = len(split_data)

        pitches = []
        mceps = []
        apdcs = []

        for i, wav in enumerate(split_data):
            # Compute fundamental frequency by harvest
            f0, t = pw.harvest(
                wav.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
                f0_ceil=1000,
            )

            sp = pw.cheaptrick(
                wav.astype(np.float64),
                f0,
                t,
                self.sampling_rate
            )
            mcep = pysptk.sp2mc(
                sp,
                order=self.config.mcep_and_apdc.mcep_channels,
                alpha=self.alpha
            )

            apdc = pw.d4c(
                wav.astype(np.float64),
                f0,
                t,
                self.sampling_rate,
                threshold=0.15,
            )

            if np.any(np.isnan(apdc)):
                raise RuntimeError(f"apdc has nan: {basename}, {i}")
                # apdc[np.isnan(apdc)] = 1

            apdc = pysptk.sp2mc(
                apdc,
                order=self.config.mcep_and_apdc.apdc_channels,
                alpha=self.alpha
            )

            # Compute mel-scale spectrogram for length assertion
            wav_torch = torch.FloatTensor(wav).unsqueeze(0)
            spec = spectrogram(
                y=wav_torch,
                n_fft=self.config.stft.filter_length,
                hop_size=self.config.stft.hop_length,
                win_size=self.config.stft.win_length,
            )
            spec = torch.squeeze(spec, 0).numpy().astype(np.float32)

            spec_len = spec.shape[1]
            f0 = f0[:spec_len]
            mcep = mcep[:spec_len]
            apdc = apdc[:spec_len]
            voiced_flag = f0 != 0

            if np.sum(voiced_flag) <= 1:
                tqdm.write(
                    f"W: {basename}, is not voiced"
                )

            f0 = np.log(f0 + 1e-12)

            # Save files
            wav_filename = "wav-{}_{}.npy".format(basename, i)
            np.save(
                os.path.join(self.out_dir, "wav", wav_filename),
                wav,
            )

            pitch_filename = "pitch-{}_{}.npy".format(basename, i)
            np.save(os.path.join(self.out_dir, "pitch", pitch_filename), f0)

            mcep_filename = "mcep-{}_{}.npy".format(basename, i)
            np.save(
                os.path.join(self.out_dir, "mcep", mcep_filename),
                mcep,
            )

            apdc_filename = "apdc-{}_{}.npy".format(basename, i)
            np.save(
                os.path.join(self.out_dir, "apdc", apdc_filename),
                apdc,
            )

            length += spec_len

            if spec_len > 0:
                pitches.append(f0[voiced_flag].reshape((-1, 1)))
                mceps.append(mcep.reshape((-1, 1)))
                apdcs.append(apdc.reshape((-1, 1)))

        return {
            "basename": basename,
            "length": length,
            "split_count": split_count,
            "pitches": pitches,
            "mceps": mceps,
            "apdcs": apdcs,
        }

    def normalize(
        self, in_dir: str, mean: float, std: float
    ) -> Tuple[np.generic, np.generic]:
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in tqdm(os.listdir(in_dir), desc="Normalizing"):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values.reshape(-1, 1)))
            min_value = min(min_value, min(values.reshape(-1, 1)))

        return min_value, max_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="path to preprocess.yaml")
    parser.add_argument(
        "-s",
        "--split",
        action="store_true",
        help="split wav files into smaller chunks based on silence",
    )
    args = parser.parse_args()

    config = AllConfig.load(args.config)
    preprocessor = Preprocessor(config.preprocess, split=args.split)
    preprocessor.build_from_path()
