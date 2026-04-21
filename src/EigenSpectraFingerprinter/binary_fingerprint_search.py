import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import convolve

class PCAFingerprintSearchStrategy(ABC):
    """An interface for searching a query fingerprint in a database."""

    @abstractmethod
    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        pass

class HammingSearch(PCAFingerprintSearchStrategy):
    """
    Searches using a sliding window with Hamming distance.
    Use with binary fingerprints.
    """

    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        # Ensure fingerprints are bools
        Gamma_Q = query_fingerprint.astype(bool)
        M_Q, phi = Gamma_Q.shape
        norm_const = M_Q * phi

        # Initialize values
        D_H_min_best = float('inf')
        s_hat = None

        # Iterate through every track in the database
        for s, Gamma_s in db.items():
            Gamma_s = Gamma_s.astype(bool)
            M_s = Gamma_s.shape[0]

            # Skip tracks that are shorter than the query
            if M_Q > M_s:
                continue

            # Iterate through the set of possible offsets
            D_H_min = float('inf')
            for delta in range(M_s - M_Q + 1):

                # Extract the segment of the track fingerprint to compare against
                Gamma_s_sub = Gamma_s[delta: delta + M_Q]

                # Sum up all the bit mismatches
                D_H = np.sum(Gamma_Q != Gamma_s_sub)

                # Minimize the distance over all offsets
                if D_H < D_H_min:
                    D_H_min = D_H

            # Track argmin over all songs
            if D_H_min < D_H_min_best:
                D_H_min_best = D_H_min
                s_hat = s

        # If no track was checked, return null
        if s_hat is None:
            return None, float('inf')

        # Normalize the distance
        D_bar_H_min = D_H_min_best / norm_const
        return s_hat, D_bar_H_min

class HammingSearchOptimized(PCAFingerprintSearchStrategy):
    """
    Searches using a sliding window with Hamming distance.
    Use with binary fingerprints.
    """

    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        # Ensure fingerprints are bools
        Gamma_Q = query_fingerprint.astype(bool)
        M_Q, phi = Gamma_Q.shape
        Q_packed = np.packbits(Gamma_Q, axis=1)
        norm_const = M_Q * phi

        # Initialize values
        s_hat = None
        D_min_best = np.iinfo(np.int64).max

        # Iterate through every track in the database
        for s, Gamma_s in db.items():
            Gamma_s = Gamma_s.astype(bool)
            M_s = Gamma_s.shape[0]

            # Skip tracks that are shorter than the query
            if M_Q > M_s:
                continue

            S_packed = np.packbits(Gamma_s.astype(bool), axis=1)

            sliding_window = np.lib.stride_tricks.sliding_window_view(
                S_packed, (M_Q, S_packed.shape[1])
            ).squeeze(axis=1)
            xor = sliding_window ^ Q_packed

            D_all = np.bitwise_count(xor).sum(axis=(1, 2))
            D_min = D_all.min()
            if D_min < D_min_best:
                s_hat = s
                D_min_best = int(D_min)


        # If no track was checked, return null
        if s_hat is None:
            return None, float('inf')

        # Normalize the distance
        D_bar_min = D_min_best / norm_const
        return s_hat, D_bar_min

class FFTConvolveSearch(PCAFingerprintSearchStrategy):
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