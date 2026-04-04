from abc import ABC, abstractmethod

import numpy as np
from pedalboard import Reverb, Distortion, Clipping, PitchShift


class AudioTestCase(ABC):
    """Abstract base class for a test case that applies a transformation to an audio signal."""

    def __init__(self, **kwargs):
        self.params = kwargs

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"

    @abstractmethod
    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        """Applies the transformation to the audio signal."""
        pass


class CleanTrackTest(AudioTestCase):
    """Control case - no transformation is applied."""

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return audio


class WhiteNoiseTest(AudioTestCase):
    """Adds Gaussian white noise to the audio."""

    def __init__(self, snr_db: int = 5):
        super().__init__(snr_db=snr_db)

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        average_signal_power = np.mean(audio ** 2) + 1e-12
        target_average_white_noise_power = average_signal_power / (10 ** (self.params['snr_db'] / 10))

        white_noise_standard_deviation = np.sqrt(target_average_white_noise_power)
        white_noise_samples = rng.normal(loc=0, scale=white_noise_standard_deviation, size=audio.shape[0]).astype(
            audio.dtype)

        noisy_audio_samples = audio + white_noise_samples
        return noisy_audio_samples
        # return np.clip(noisy_audio_samples, -1, 1)


class PitchShiftTest(AudioTestCase):
    """Shifts the pitch of the audio."""

    def __init__(self, n_steps: float = 1.5):
        super().__init__(n_steps=n_steps)

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return PitchShift(semitones=self.params['n_steps'])(audio, sr)


class ReverbTest(AudioTestCase):
    """Adds reverb to the audio."""

    def __init__(self, room_size: float, damping: float, wet_level: float, dry_level: float):
        super().__init__(room_size=room_size, damping=damping, wet_level=wet_level, dry_level=dry_level)

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return Reverb(room_size=self.params['room_size'],
                      damping=self.params['damping'],
                      wet_level=self.params['wet_level'],
                      dry_level=self.params['dry_level'])(audio, sr)


class DistortionTest(AudioTestCase):
    """Adds distortion to the audio."""

    def __init__(self, drive_db: int):
        super().__init__(drive_db=drive_db)

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return Distortion(drive_db=self.params['drive_db'])(audio, sr)


class ClippingDbTest(AudioTestCase):
    """Clips the decibel max to `threshold_db`."""

    def __init__(self, threshold_db: int):
        super().__init__(threshold_db=threshold_db)

    def apply(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return Clipping(threshold_db=self.params['threshold_db'])(audio, sr)
