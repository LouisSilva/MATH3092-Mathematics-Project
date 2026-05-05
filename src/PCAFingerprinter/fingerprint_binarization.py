from abc import ABC, abstractmethod
import numpy as np

class FingerprintProjectionStrategy(ABC):
    """An interface for converting a projected matrix into a fingerprint."""

    @abstractmethod
    def generate(self, P: np.ndarray) -> np.ndarray:
        pass


class DeltaFingerprint(FingerprintProjectionStrategy):
    """Generates a binary fingerprint using temporal difference (delta features)."""

    def generate(self, P: np.ndarray) -> np.ndarray:
        Delta_P = P[1:] - P[:-1]
        Gamma = (Delta_P > 0).astype(np.uint8)

        # Instead of storing the binary fingerprint as an array of bytes,
        # we compress every sequence of 8 boolean values into a single unsigned
        # 8-bit integer (uint8), thereby reducing space complexity by a factor of 8
        return np.packbits(Gamma.astype(bool), axis=1)
