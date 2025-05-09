import torch
from torch.utils.data import Dataset
import torchaudio
import random
import numpy as np


class SpeakerAudioDataset(Dataset):
    def __init__(self, data_list, speaker_to_id, target_sr, transform=None):
        """
        Args:
            data_list (list): List of tuples (audio_path, json_path, speaker_name).
            speaker_to_id (dict): Mapping from speaker name to integer ID.
            target_sr (int): Target sample rate.
        """
        self.data_list = data_list
        self.speaker_to_id = speaker_to_id
        self.target_sr = target_sr
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # We don't need json_path here anymore
        audio_path, _, speaker_name = self.data_list[idx]
        speaker_id = self.speaker_to_id[speaker_name]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Remove channel dimension, should be (seq_len,)
        waveform = waveform.squeeze(0)

        # Ensure waveform is not empty after processing
        if waveform.numel() == 0:
            # print(f"Warning: Empty waveform for {audio_path}, skipping (returning None).")
            return None

        if self.transform:
            waveform_np = waveform.numpy()
            waveform_np = self.transform(
                samples=waveform_np, sample_rate=self.target_sr)
            waveform = torch.tensor(waveform_np)

        return waveform, speaker_id


class DynamicSpeakerSegmentDataset(torch.utils.data.Dataset):
    """
    Each __getitem__ builds one 4-second waveform by concatenating
    1..MAX_SEG_PER_CLIP speech segments of the same speaker.
    """

    def __init__(self, manifest, speaker_to_id,
                 target_sr=16000, target_len=4, samples_per_spk=100, max_segments=4, min_clip_percentage=30, max_clip_percentage=60, transform=None):
        self.manifest = manifest
        self.speaker_names = list(manifest.keys())
        self.spk_to_id = speaker_to_id
        self.sr = target_sr
        self.target_len = target_len*target_sr
        self.max_segments = max_segments
        self.transform = transform
        self.min_clip_percentage = min_clip_percentage
        self.max_clip_percentage = max_clip_percentage
        # choose how many virtual samples per speaker you want;
        # here: 100 per real clip ≈ plenty of diversity
        self.samples_per_spk = samples_per_spk
        self.dataset_len = len(self.speaker_names) * self.samples_per_spk

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # pick speaker
        spk = self.speaker_names[idx % len(self.speaker_names)]
        segs = self.manifest[spk]
        random.shuffle(segs)

        clip = torch.empty(0)
        i = 0
        while len(clip) < self.target_len and i < len(segs):
            wav_path, s0, s1 = segs[i]
            i += 1
            num_frames = s1 - s0

            if num_frames <= 0:
                continue

            # partial read
            w, sr = torchaudio.load(wav_path,
                                    frame_offset=s0,
                                    num_frames=num_frames)
            if sr != self.sr:
                w = torchaudio.functional.resample(w, sr, self.sr)
            if w.shape[0] > 1:
                w = torch.mean(w, dim=0, keepdim=True)

            # If w is longer than 2 seconds, randomly extract a segment whose length is
            # between min_clip_percentage% and max_clip_percentage% of the total waveform.
            # The segment is randomly positioned but guaranteed to fit within the waveform.
            if w.shape[0] / self.sr > 2:
                l = w.shape[0]
                clip_percentage = np.random.randint(
                    self.min_clip_percentage, self.max_clip_percentage + 1)
                clip_length = l * clip_percentage // 100

                start_idx = np.random.randint(0, l - clip_length + 1)
                end_idx = start_idx + clip_length

                w = w[start_idx:end_idx]

            clip = torch.cat([clip, w.squeeze(0)])

            # stop after MAX segments
            if i >= self.max_segments:
                break
        # pad or truncate to exactly 4 s
        if len(clip) < self.target_len:
            pad_before = random.randint(0, self.target_len - len(clip))
            pad_after = self.target_len - len(clip) - pad_before
            clip = torch.nn.functional.pad(clip, (pad_before, pad_after))
        else:
            clip = clip[:self.target_len]

        if self.transform:
            clip_np = clip.numpy()
            clip_np = self.transform(samples=clip_np, sample_rate=self.sr)

        return torch.tensor(clip_np), self.spk_to_id[spk]
