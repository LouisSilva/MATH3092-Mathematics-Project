import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .spectrogram_generation import compute_spectrogram_from_file, compute_spectrogram_from_samples, load_audio
from .fingerprint_binarization import FingerprintProjectionStrategy


class PCAFingerprinter:
    def __init__(
            self,
            fingerprint_strategy: FingerprintProjectionStrategy,
            f_s: int = 44100,
            L: int = 4096,
            H: int = 2048,
            B: int = 128,
            tau: int = 80,
            window_type: str = "hann",
            gamma: int = 50000,
            kappa: int = 50,
            phi: int = 16,
            random_state: int = 42,
    ):
        self.fingerprint_strategy = fingerprint_strategy
        self.f_s = f_s
        self.L = L
        self.H = H
        self.B = B
        self.tau = tau
        self.window_type = window_type
        self.gamma = gamma
        self.kappa = kappa
        self.phi = phi
        self.random_state = random_state

        self.is_fitted = False
        self.model = PCA(n_components=phi, random_state=random_state)

    def train(self, file_list: list[str]) -> None:
        """
        Fits the PCA basis using a random subset of segments from the training files.
        :arg file_list: List of audio files used to create the training matrix.
        """
        training_segments = []
        total_segments = 0

        # Shuffle the file list to avoid biasing towards the files at the beginning of the dataset
        np.random.shuffle(file_list)

        # Calculate the estimated number of files that will be used to create the matrix, for the tqdm loading bar
        rho = len(file_list)
        estimated_num_files = min(rho * self.kappa, self.gamma) // self.kappa

        for file_path in tqdm(file_list, total=estimated_num_files, desc="Creating training data matrix for PCA"):
            # Only allow a max of gamma rows
            segments_remaining_before_max_size = self.gamma - total_segments
            if segments_remaining_before_max_size <= 0:
                print(f"Reached the limit of gamma={self.gamma} training segments.")
                break

            # Select only a maximum of kappa segments
            spectrogram = compute_spectrogram_from_file(
                file_path,
                f_s=self.f_s,
                L=self.L,
                H=self.H,
                B=self.B,
                window_type=self.window_type,
                tau=self.tau
            )

            max_allowed_segments = min(self.kappa, segments_remaining_before_max_size)
            if spectrogram.shape[0] > max_allowed_segments:
                indices = np.random.choice(spectrogram.shape[0], max_allowed_segments, replace=False)
                spectrogram = spectrogram[indices]

            # Append these segments to the training matrix
            training_segments.append(spectrogram)
            total_segments += spectrogram.shape[0]

        # Vertically stack all segments: (M_t, B)
        T = np.vstack(training_segments)

        print(f"Fitting model on training matrix with shape: {T.shape}...")
        self.model.fit(T)  # The model centres itself for us
        self.is_fitted = True

        explained_variance = self.model.explained_variance_ratio_.sum()
        print(f"Model fitted, explained variance: {explained_variance:.4f}.")

    def save_model(self, filepath: str) -> None:
        """Serializes the trained model and its hyperparameters to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before it can be saved to disk.")

        state = {
            "hyperparameters": {
                "f_s": self.f_s,
                "L": self.L,
                "H": self.H,
                "B": self.B,
                "tau": self.tau,
                "window_type": self.window_type,
                "gamma": self.gamma,
                "kappa": self.kappa,
                "phi": self.phi,
                "random_state": self.random_state
            },
            "fingerprint_strategy": self.fingerprint_strategy,
            "model": self.model,
        }

        joblib.dump(state, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "PCAFingerprinter":
        """Factory method to load a trained model from disk and reconstruct its configuration."""
        state = joblib.load(filepath)

        instance = cls(fingerprint_strategy=state["fingerprint_strategy"], **state["hyperparameters"])
        instance.model = state["model"]
        instance.is_fitted = True
        return instance

    def get_fingerprint_from_file(self, filepath: str) -> np.ndarray:
        """
        Generates a fingerprint.
        :arg filepath: The filepath to the audio to be fingerprinted.
        :returns: Low-dimensional representation of the given audio's spectrogram, a matrix with dimensions (``M``, ``phi``).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before fingerprinting.")

        audio, sr = load_audio(filepath, self.f_s)
        return self.get_fingerprint_from_samples(audio)

    def get_fingerprint_from_samples(self, audio: np.ndarray) -> np.ndarray:
        """
        Generates a fingerprint.
        :arg audio: The numpy array of samples to fingerprint.
        :returns: Low-dimensional representation of the given audio's spectrogram, a matrix with dimensions (``M``, ``phi``).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before fingerprinting.")

        spectrogram = compute_spectrogram_from_samples(audio, self.L, self.H, self.B, self.tau, self.window_type)
        compressed = self.model.transform(spectrogram)
        return self.fingerprint_strategy.generate(compressed)

    def plot_scree(self, cumulative: bool = True) -> None:
        """
        Displays a scree plot of explained variance.
        :arg cumulative: Whether to display the cumulative explained variance.
        """
        if not self.is_fitted:
            raise RuntimeError("SVD must be trained before plotting scree plot.")

        evr = self.model.explained_variance_ratio_
        x = np.arange(1, len(evr) + 1)

        plt.figure(figsize=(7, 4))
        plt.plot(x, evr, marker="o", label="Per-component")

        if cumulative:
            plt.plot(x, np.cumsum(evr), marker="s", label="Cumulative")

        plt.xlabel("Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("SVD Scree Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
