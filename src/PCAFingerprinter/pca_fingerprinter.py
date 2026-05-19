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
            L: int = 2048,
            H: int = 1024,
            B: int = 128,
            window_type: str = "hann",
            tau: int = 80,
            kappa: int = 50,
            gamma: int = 50000,
            phi: int = 16,
            random_state: int = 42,
    ):
        self.fingerprint_strategy = fingerprint_strategy
        self.f_s = f_s
        self.L = L
        self.H = H
        self.B = B
        self.window_type = window_type
        self.tau = tau
        self.kappa = kappa
        self.gamma = gamma
        self.phi = phi
        self.random_state = random_state

        self.is_fitted = False
        self.model = PCA(n_components=phi, random_state=random_state)
        self.rng = np.random.default_rng(random_state)

    def train(self, file_list: list[str]) -> None:
        """
        Fits the PCA basis using a random subset of segments from the training files.
        :arg file_list: List of audio files used to create the training matrix.
        """
        T_rows = []
        M_kappa = 0

        # Shuffle the file list to avoid biasing towards the files at the beginning of the dataset
        files = self.rng.permutation(file_list)

        # Calculate the estimated number of files that will be used to create the matrix, for the tqdm loading bar
        rho = len(files)
        estimated_num_files = min(rho * self.kappa, self.gamma) // self.kappa

        for file_path in tqdm(files, total=estimated_num_files, desc="Creating training data matrix for PCA"):
            # Only allow a max of gamma rows
            gamma_remaining = self.gamma - M_kappa
            if gamma_remaining <= 0:
                print(f"Reached the limit of gamma={self.gamma} training segments.")
                break

            # Select only a maximum of kappa segments
            S = compute_spectrogram_from_file(
                file_path,
                f_s=self.f_s,
                L=self.L,
                H=self.H,
                B=self.B,
                window_type=self.window_type,
                tau=self.tau
            )

            per_song_cap = min(self.kappa, gamma_remaining)
            if S.shape[0] > per_song_cap:
                I_s = self.rng.choice(S.shape[0], per_song_cap, replace=False)
                S = S[I_s]

            # Append these segments to the training matrix
            T_rows.append(S)
            M_kappa += S.shape[0]

        # Vertically stack all segments: (M_t, B)
        T = np.vstack(T_rows)

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
                "window_type": self.window_type,
                "tau": self.tau,
                "kappa": self.kappa,
                "gamma": self.gamma,
                "phi": self.phi,
                "random_state": self.random_state
            },
            "fingerprint_strategy": self.fingerprint_strategy,
            "model": self.model,
        }

        joblib.dump(state, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "PCAFingerprinter":
        """Factory method to load a trained model from disk and load its configuration."""
        state = joblib.load(filepath)

        instance = cls(fingerprint_strategy=state["fingerprint_strategy"], **state["hyperparameters"])
        instance.model = state["model"]
        instance.is_fitted = True
        return instance

    def get_fingerprint_from_file(self, filepath: str) -> np.ndarray:
        """
        Generates a fingerprint.
        :arg filepath: Filepath of the audio to be fingerprinted.
        :returns: Low-dimensional representation of the given audio's sample vector, a matrix with dimensions (``M``, ``phi``).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before fingerprinting.")

        x, _ = load_audio(filepath, self.f_s)
        return self.get_fingerprint_from_samples(x)

    def get_fingerprint_from_samples(self, x: np.ndarray) -> np.ndarray:
        """
        Generates a fingerprint.
        :arg x: NumPy array of samples to fingerprint.
        :returns: Low-dimensional representation of the given audio's sample vector, a matrix with dimensions (``M``, ``phi``).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before fingerprinting.")

        S = compute_spectrogram_from_samples(
            x=x,
            f_s=self.f_s,
            L=self.L,
            H=self.H,
            B=self.B,
            window_type=self.window_type,
            tau=self.tau
        )

        P = self.model.transform(S)
        Gamma = self.fingerprint_strategy.generate(P)
        return Gamma

    def plot_scree(self, cumulative: bool = True) -> None:
        """
        Displays a scree plot of explained variance.
        :arg cumulative: Whether to display the cumulative explained variance.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before plotting scree plot.")

        explained_variance = self.model.explained_variance_ratio_
        x_axis = np.arange(1, len(explained_variance) + 1)

        plt.figure(figsize=(7, 4))
        plt.plot(x_axis, explained_variance, marker="o", label="Per-component")

        if cumulative:
            plt.plot(x_axis, np.cumsum(explained_variance), marker="s", label="Cumulative")

        plt.xlabel("Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
