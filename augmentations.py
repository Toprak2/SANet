from audiomentations import Compose
import numpy as np
from audiomentations import Compose, ApplyImpulseResponse, \
    Gain, TimeStretch, ClippingDistortion, PitchShift
from audiomentations.core.transforms_interface import BaseWaveformTransform


class RMSNormalize(BaseWaveformTransform):
    """
    Scale each clip so its RMS equals target_rms (linear, not dB).
    """

    def __init__(self, target_rms: float = 0.1, eps: float = 1e-9, p: float = 1.0):
        super().__init__(p)
        self.target_rms = target_rms
        self.eps = eps

    def apply(self, samples: np.ndarray, sample_rate: int):
        rms = np.sqrt(np.mean(samples ** 2) + self.eps)
        gain = self.target_rms / rms
        return samples * gain


def get_augmentations():
    train_augs = Compose([
        # 1) waveform‑level augments that keep raw amplitude
        # ApplyImpulseResponse(ir_path="/content/MIT_IR", p=0.4),
        TimeStretch(min_rate=0.97, max_rate=1.03, p=0.3),
        PitchShift(min_semitones=-0.25, max_semitones=0.25, p=0.1),
        # 2) loudness equalisation
        RMSNormalize(target_rms=0.1, p=1.0),
        # 3) random gain to teach invariance (train‑only)
        Gain(min_gain_db=-12, max_gain_db=12, p=0.8),
    ], shuffle=False)

    val_augs = Compose([
        # per‑clip loudness equalisation
        RMSNormalize(target_rms=0.10, p=1.0),
    ], shuffle=False)

    return train_augs, val_augs
