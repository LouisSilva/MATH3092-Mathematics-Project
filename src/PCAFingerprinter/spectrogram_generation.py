import numpy as np
import librosa

def load_audio(
        file_path: str,
        f_s: int = 44100,
        offset: float = 0.0,
        duration: float | None = None
) -> tuple[np.ndarray, int]:
    """
    Loads mono audio at a target sample rate ``f_s``.

    Optionally, it can load a slice of the audio using a start ``offset`` and ``duration``.
    Time values are specified in seconds, and frequency values are specified in hertz.

    :arg file_path: Path to the audio file.
    :arg f_s: The sampling rate to use for loading the audio.
    :arg offset: Start time (seconds) to begin loading from.
    :arg duration: Maximum duration (seconds) of the audio to load. If ``None``, then it will load until the end of the file.
    :returns: A tuple ``(x, f_s_returned)`` where ``x`` is the audio array and ``f_s_returned`` is the sampling rate.
    """
    x, f_s_returned = librosa.load(file_path, sr=f_s, mono=True, offset=offset, duration=duration)
    return x, f_s_returned


def compute_spectrogram_from_samples(
        x: np.ndarray,
        f_s: int = 44100,
        L: int = 2048,
        H: int = 1024,
        B: int = 128,
        window_type: str = "hann",
        tau: int = 80,
) -> np.ndarray:
    """
    Computes a spectrogram from already-loaded audio ``x``.

    :arg x: 1-D array of samples.
    :arg f_s: The sampling rate to use for loading the audio.
    :arg L: Number of samples in each segment.
    :arg H: Sample offset between adjacent segments (hop size).
    :arg B: Number of Mel bins to use.
    :arg window_type: Window function used in the spectrogram computation.
    :arg tau: The decibel limit.
    :returns: A spectrogram with shape ``(M, B)`` where the values lie in the range ``[0, 1]``.
    """
    S_mel = librosa.feature.melspectrogram(
        y=x,
        sr=f_s,
        n_fft=L,
        hop_length=H,
        n_mels=B,
        window=window_type,
        power=2
    )

    # Convert to dB
    S_db = librosa.power_to_db(S_mel, ref=np.max, top_db=tau)

    # Map [-top_db, 0] to [0, 1]
    S = (S_db + tau) / tau

    # Clip to ensure 0-1 bounds
    return np.clip(S, 0, 1).T


def compute_spectrogram_from_file(
        file_path: str,
        f_s: int = 44100,
        L: int = 2048,
        H: int = 1024,
        B: int = 128,
        window_type: str = "hann",
        tau: int = 80
) -> np.ndarray:
    """
    Loads an audio file and computes its spectrogram.

    :arg file_path: Path to the audio file.
    :arg f_s: The sampling rate to use for loading the audio.
    :arg L: Number of samples in each segment.
    :arg H: Sample offset between adjacent segments (hop size).
    :arg B: Number of Mel bins to use.
    :arg window_type: Window function used in the spectrogram computation.
    :arg tau: The decibel limit.
    :returns: A spectrogram with shape ``(M, B)`` where the values lie in the range ``[0, 1]``.
    """
    x, _ = load_audio(file_path, f_s=f_s)
    return compute_spectrogram_from_samples(
        x,
        f_s=f_s,
        L=L,
        H=H,
        B=B,
        window_type=window_type,
        tau=tau
    )