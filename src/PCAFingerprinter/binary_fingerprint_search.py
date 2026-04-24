import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import convolve

class PCAFingerprintSearchStrategy(ABC):
    """An interface for searching a query fingerprint in a database."""

    @abstractmethod
    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        pass

class HammingSearchSlow(PCAFingerprintSearchStrategy):
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

class HammingSearch(PCAFingerprintSearchStrategy):
    """
    Searches using a sliding window with Hamming distance.
    Use with binary fingerprints.
    """

    def search(self, query_fingerprint: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str | None, float]:
        # Ensure fingerprints are bools
        Gamma_Q = query_fingerprint.astype(bool)

        # Get dimensions and calculate the normalization constant
        M_Q, phi = Gamma_Q.shape
        norm_const = M_Q * phi

        # Instead of storing the binary fingerprint as an array of bytes,
        # we compress every sequence of 8 boolean values into a single unsigned
        # 8-bit integer (uint8), thereby reducing space complexity by a factor of 8
        Gamma_Q_packed = np.packbits(Gamma_Q, axis=1)

        # Initialize default values
        s_hat = None
        D_min_best = np.iinfo(np.int64).max

        # Iterate through every song in the database
        for s, Gamma_s in db.items():
            Gamma_s = Gamma_s.astype(bool)
            M_s = Gamma_s.shape[0]

            # Skip songs that are shorter than the query
            if M_Q > M_s:
                continue

            Gamma_s_packed = np.packbits(Gamma_s.astype(bool), axis=1)

            # Instead of allocating memory and copying each
            # Gamma_s[delta : delta + M_Q - 1, :] sub matrix to a new array,
            # we use an optimized function coded in C which doesn't copy any data
            sliding_window = np.lib.stride_tricks.sliding_window_view(
                Gamma_s_packed, (M_Q, Gamma_s_packed.shape[1])
            ).squeeze(axis=1)

            # Apply the XOR operator
            xor = sliding_window ^ Gamma_Q_packed

            # Count how many bits are set to 1
            D_all = np.bitwise_count(xor).sum(axis=(1, 2))

            # Apply min and argmin
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
    Search using 1D FFT Convolution.
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