from abc import ABC, abstractmethod
import numpy as np

class FingerprintProjectionStrategy(ABC):
    """An interface for converting a projected matrix into a fingerprint."""

    @abstractmethod
    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        pass


class DeltaFingerprint(FingerprintProjectionStrategy):
    """Generates a binary fingerprint using temporal difference (delta features)."""

    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        Delta_P = np.zeros_like(projected_matrix)

        # Subtract previous row from current row
        Delta_P[1:] = projected_matrix[1:] - projected_matrix[:-1]

        # Create delta feature
        Gamma = (Delta_P > 0).astype(np.uint8)

        return Gamma


class MedianThresholdingFingerprint(FingerprintProjectionStrategy):
    """Generates a binary fingerprint using median-thresholding."""

    def generate(self, projected_matrix: np.ndarray) -> np.ndarray:
        # Find the median of each row
        medians = np.median(projected_matrix, axis=1, keepdims=True)

        # Compare each component to the median of its segment
        binary_fingerprint = projected_matrix > medians
        return binary_fingerprint


