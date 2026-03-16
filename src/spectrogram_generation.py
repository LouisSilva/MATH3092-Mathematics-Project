import numpy as np
import librosa

def load_audio(
        file_path: str,
        f_s: int = 22050,
        offset: float = 0.0,
        duration: float | None = None
) -> tuple[np.ndarray, int]:
    """
    Loads mono audio at a target sample rate.

    Optionally, it can load a slice of the audio using a start offset and duration.
    Time values are specified in seconds, and frequency values are specified in hertz.

    Args:
        file_path: Path to the audio file.
        f_s: The sampling rate to use for loading the audio.
        offset: Start time (seconds) to begin loading from.
        duration: Maximum duration (seconds) of the audio to load. If None,
            then it will load until the end of the file.

    Returns:
        A tuple ``(x, sr)`` where ``x`` is the audio array and ``sr`` is the sampling rate.
    """
    x, sr = librosa.load(file_path, sr=f_s, mono=True, offset=offset, duration=duration)
    return x, sr


def compute_spectrogram_from_audio(
        audio: np.ndarray,
        L: int = 2048,
        H: int = 1024,
        F: int = 128,
        tau: int = 80,
        window_type: str = "hann"
) -> np.ndarray:
    """
    Computes a spectrogram from already-loaded audio.

    Args:
        audio: 1-D array of samples.
        L: Number of samples in each segment.
        H: Sample offset between adjacent segments (hop size).
        F: Number of Mel bins to use.
        tau: The decibel limit.
        window_type: Window function used in the spectrogram computation.

    Returns:
        A spectrogram with shape ``(M, F)``
        where the values lie in the range [0, 1].
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_fft=L, hop_length=H, n_mels=F, window=window_type)
    power = np.abs(mel_spectrogram) ** 2

    # Convert to dB
    mel_db_spectrogram = librosa.power_to_db(power, ref=np.max, top_db=tau)

    # Map [-top_db, 0] to [0, 1]
    mel_db_spectrogram_scaled = (mel_db_spectrogram + tau) / tau

    # Clip to ensure 0-1 bounds
    return np.clip(mel_db_spectrogram_scaled, 0, 1).T


def compute_spectrogram(
        file_path: str,
        f_s: int = 22050,
        L: int = 2048,
        H: int = 1024,
        F: int = 128,
        tau: int = 80,
        window_type: str = "hann"
) -> np.ndarray:
    """
    Loads an audio file and computes its spectrogram.

    Args:
        file_path: Path to the audio file.
        f_s: The sampling rate to use for loading the audio.
        L: Number of samples in each segment.
        H: Sample offset between adjacent segments (hop size).
        F: Number of Mel bins to use.
        tau: Decibel limit.
        window_type: Window function used in the spectrogram computation.

    Returns:
        A spectrogram with shape ``(M, F)``
        where the values lie in the range [0, 1].
    """
    audio, _ = load_audio(file_path, f_s=f_s)
    return compute_spectrogram_from_audio(audio, L=L, H=H, F=F, tau=tau, window_type=window_type)