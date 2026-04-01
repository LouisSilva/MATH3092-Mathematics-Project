import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import convolve

class FingerprintSearchStrategy(ABC):
    """An interface for searching a query fingerprint in a database."""

    @abstractmethod
    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        pass

class HammingSearch(FingerprintSearchStrategy):
    """
    Searches using a sliding window with Hamming distance.
    Use with binary fingerprints.
    """

    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        # Ensure fingerprints are bools
        query_fp = query_fingerprint.astype(bool)
        query_len, n_features = query_fp.shape
        total_bits = query_len * n_features

        # Initialize values
        best_min_dist = float('inf')
        best_track = None

        # Iterate through every track in the database
        for track_id, track_fingerprint in db.items():
            track_fp = track_fingerprint.astype(bool)
            track_len = track_fp.shape[0]

            # Skip tracks that are shorter than the query
            if query_len > track_len:
                continue

            # Iterate through the set of possible offsets
            for offset in range(track_len - query_len + 1):

                # Extract the segment of the track fingerprint to compare against
                sub_matrix = track_fp[offset: offset + query_len]

                # Sum up all the bit mismatches
                current_dist = np.sum(query_fp != sub_matrix)

                # Minimize the distance
                if current_dist < best_min_dist:
                    best_min_dist = current_dist
                    best_track = track_id

        # If no track was checked, return null
        if best_track is None:
            return None, float('inf')

        # Normalize the best distance
        normalized_min_dist = best_min_dist / total_bits

        return best_track, normalized_min_dist

class FFTConvolveSearch(FingerprintSearchStrategy):
    """
    Extremely fast search using 1D FFT Convolution.
    Flattens the 2D fingerprints and computes correlation in one go.
    """

    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        # query shape: (Q, F)
        Q, F = query_fingerprint.shape
        total_bits = Q * F

        # Pre-process Query
        # Convert {0, 1} -> {-1, 1}
        query_bipolar = 2 * query_fingerprint.astype(np.float32) - 1

        # FLATTEN to 1D
        query_flat = query_bipolar.flatten()

        # REVERSE for convolution -> correlation equivalence
        query_flat = query_flat[::-1]

        best_dist = float('inf')
        best_track = None

        for track_id, track_fingerprint in db.items():
            T, _ = track_fingerprint.shape

            if Q > T:
                continue

            # Pre-process Track
            track_bipolar = 2 * track_fingerprint.astype(np.float32) - 1
            track_flat = track_bipolar.flatten()

            # FFT Convolution
            # mode='valid' returns only overlaps where query fits inside track
            correlation = convolve(track_flat, query_flat, mode='valid', method='fft')

            # Extract valid alignments. Because we flattened, 'valid' convolution produces results for shifts of 1 element, 2 elements, etc.
            # We only care about shifts of exactly F (one whole time step).
            # We take every F-th element.
            valid_correlations = correlation[::F]

            if valid_correlations.size == 0:
                continue

            # Calculate Distances
            max_corr = np.max(valid_correlations)
            min_hamming_dist = (total_bits - max_corr) / 2.0
            normalized_dist = min_hamming_dist / total_bits

            if normalized_dist < best_dist:
                best_dist = normalized_dist
                best_track = track_id

        return best_track, best_dist