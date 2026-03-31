import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .spectrogram_generation import compute_spectrogram

class FingerprintStrategy(ABC):
    """An interface for converting a projected matrix into a fingerprint."""
    @abstractmethod
    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        pass

class BinaryFingerprint(FingerprintStrategy):
    """
    Generates a binary fingerprint using median-thresholding.
    """
    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        # Find the median of each row
        medians = np.median(projected_matrix, axis=1, keepdims=True)

        # Compare each component to the median of its segment
        binary_fingerprint = projected_matrix > medians
        return binary_fingerprint

class DeltaFingerprint(FingerprintStrategy):
    """
    Generates a binary fingerprint using temporal difference (delta features).
    """
    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        # Subtract previous row from current row
        delta = projected_matrix[1:] - projected_matrix[:-1]

        # Binarize based on the sign
        binary_fingerprint = (delta > 0).astype(int)

        return binary_fingerprint


class SVDFingerprinter:
    def __init__(
            self,
            fingerprint_strategy: FingerprintStrategy,
            gamma: int = 50000,
            kappa: int = 50,
            phi: int = 64,
            f_s: int = 44100,
            L: int = 2048,
            H: int = 1024,
            B: int = 128,
            tau: int = 80,
            window_type: str = "hann",
            random_state: int = 42,
    ):
        self.fingerprint_strategy = fingerprint_strategy
        self.gamma = gamma
        self.kappa = kappa
        self.phi = phi
        self.f_s = f_s
        self.L = L
        self.H = H
        self.B = B
        self.tau = tau
        self.window_type = window_type

        self.is_fitted = False
        self.model = PCA(n_components=phi, random_state=random_state)

    def train(self, file_list: list[str]):
        """
        Fits the SVD basis using a random subset of segments from the training files.
        """
        num_training_files = min(len(file_list) * self.kappa, self.gamma)
        training_segments = []
        total_rows = 0

        print(f"Creating training data matrix for SVD...")
        np.random.shuffle(file_list)

        for file_path in tqdm(file_list, total=num_training_files):
            # Only allow a max of gamma rows
            rows_remaining = self.gamma - total_rows
            if rows_remaining <= 0:
                print(f"Reached the limit of gamma={self.gamma} training segments.")
                break

            # Select only a maximum of kappa segments
            spectrogram = self.__compute_spectrogram(file_path)
            max_allowed_segments = min(self.kappa, rows_remaining)
            if spectrogram.shape[0] > max_allowed_segments:
                indices = np.random.choice(spectrogram.shape[0], max_allowed_segments, replace=False)
                spectrogram = spectrogram[indices]

            # Append these segments to the training matrix
            training_segments.append(spectrogram)
            total_rows += spectrogram.shape[0]

        # Vertically stack all segments: (gamma, B)
        T = np.vstack(training_segments)

        print(f"Fitting model on training matrix with shape: {T.shape}...")
        self.model.fit(T)  # The model centres itself for us
        self.is_fitted = True

        explained_variance = self.model.explained_variance_ratio_.sum()
        print(f"Model fitted, explained variance: {explained_variance:.4f}.")

    def get_fingerprint(self, file_path: str) -> np.ndarray:
        """
        Generates a fingerprint using the configured model and strategy.
        Returns: np.ndarray: Low-rank matrix (M, phi).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before fingerprinting.")

        spectrogram = self.__compute_spectrogram(file_path)
        compressed = self.model.transform(spectrogram)
        return self.fingerprint_strategy.generate(compressed)

    def plot_scree(self, cumulative: bool = True):
        """
        Displays a scree plot of explained variance.
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

    def __compute_spectrogram(self, file_path: str) -> np.ndarray:
        return compute_spectrogram(file_path, f_s=self.f_s, L=self.L, H=self.H, B=self.B, tau=self.tau,
                                   window_type=self.window_type)