import librosa
import numpy as np
from abc import ABC, abstractmethod
from pedalboard import Reverb, Distortion, Clipping

class AudioTestCase(ABC):
    """Abstract base class for a test case."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"

    @abstractmethod
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applies the distortion to the audio signal."""
        pass

class CleanTrackTest(AudioTestCase):
    """Control case, no distortion is applied."""
    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio

class WhiteNoiseTest(AudioTestCase):
    """Adds Gaussian white noise to the audio."""
    def __init__(self, amplitude: float = 0.05):
        super().__init__(amplitude=amplitude)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        noise = np.random.normal(0, 1, size=audio.shape[0]).astype(audio.dtype)
        return audio + (self.params['amplitude'] * noise)

class PitchShiftTest(AudioTestCase):
    """Shifts the pitch of the audio."""
    def __init__(self, n_steps: float = 1.5):
        super().__init__(n_steps=n_steps)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=self.params['n_steps'])

class ReverbTest(AudioTestCase):
    """Adds reverb to the audio."""
    def __init__(self, room_size: float, damping: float, wet_level: float, dry_level: float):
        super().__init__(room_size=room_size, damping=damping, wet_level=wet_level, dry_level=dry_level)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return Reverb(room_size=self.params['room_size'],
                          damping=self.params['damping'],
                          wet_level=self.params['wet_level'],
                          dry_level=self.params['dry_level'])(audio, sr)

class DistortionTest(AudioTestCase):
    """Adds distortion to the audio."""
    def __init__(self, drive_db: int):
        super().__init__(drive_db=drive_db)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return Distortion(drive_db=self.params['drive_db'])(audio, sr)

class ClippingDbTest(AudioTestCase):
    """Clips the decibel max to `threshold_db`."""
    def __init__(self, threshold_db: int):
        super().__init__(threshold_db=threshold_db)

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return Clipping(threshold_db=self.params['threshold_db'])(audio, sr)