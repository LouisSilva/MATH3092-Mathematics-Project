import os
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod

from src.TestSuite.data_structures import MatchOutcome
from vendor.audfprint import audfprint_analyze, audfprint_match, hash_table

class RetrievalBackend(ABC):
    """
    An interface for retrieval systems.

    A backend owns:
    - optional training
    - database/index creation
    - query search
    - threshold interpretation
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @property
    @abstractmethod
    def score_name(self) -> str:
        pass

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        pass

    @property
    @abstractmethod
    def decision_rule(self) -> str:
        """Human-readable explanation of what counts as a confident match."""
        pass

    def train(self, file_list: list[str]) -> None:
        """
        Optional training hook.

        :arg file_list: List of audio files used to train on.
        """
        return None

    @abstractmethod
    def add_tracks(self, track_files: list[str]) -> None:
        pass

    @abstractmethod
    def search(self, query_path: str) -> MatchOutcome:
        pass

    @abstractmethod
    def is_confident_match(self, match_outcome: MatchOutcome) -> bool:
        """ Returns ``True`` if this result is confident enough to be accepted as a match."""
        pass


class PCARetrievalBackend(RetrievalBackend):
    backend_name = "PCA"
    score_name = "distance"
    higher_is_better = False

    def __init__(self, fingerprinter, search_strategy, confident_match_threshold: float = 0.35):
        self.fingerprinter = fingerprinter
        self.search_strategy = search_strategy
        self.confident_match_threshold = float(confident_match_threshold)
        self.db: dict[str, np.ndarray] = {}

    @property
    def decision_rule(self) -> str:
        return f"accept if distance <= {self.confident_match_threshold:.4f}"

    def train(self, file_list: list[str]) -> None:
        """
        Train the PCA model if it is not already fitted.

        :arg file_list: List of audio files used to train on.
        """
        if getattr(self.fingerprinter, "is_fitted", False):
            return

        if not file_list:
            raise ValueError("PCARetrievalBackend.train() was called with no training files.")

        self.fingerprinter.train(file_list)

    def add_tracks(self, track_files: list[str]) -> None:
        print(f"Indexing {len(track_files)} tracks with PCA backend...")

        for file_path in tqdm(track_files, total=len(track_files)):
            try:
                fingerprint = self.fingerprinter.get_fingerprint(file_path)
                track_id = os.path.basename(file_path)
                self.db[track_id] = fingerprint
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    def search(self, query_path: str) -> MatchOutcome:
        query_fingerprint = self.fingerprinter.get_fingerprint(query_path)
        best_track, best_score = self.search_strategy.search(query_fingerprint, self.db)

        return MatchOutcome(
            predicted_track_id=best_track,
            score=float("inf") if best_track is None else float(best_score),
            score_name=self.score_name,
            higher_is_better=self.higher_is_better,
        )

    def is_confident_match(self, result: MatchOutcome) -> bool:
        if result.predicted_track_id is None:
            return False

        return result.score <= self.confident_match_threshold


class ShazamRetrievalBackend(RetrievalBackend):
    """
    Shazam-style landmark/hash retrieval backend built on top of vendored ``audfprint``.

    This backend uses:
    - ``audfprint_analyze.Analyzer``
    - ``hash_table.HashTable``
    - ``audfprint_match.Matcher``
    """

    backend_name = "Shazam"
    score_name = "aligned_hashes"
    higher_is_better = True

    def __init__(
        self,
        *,
        density: float = 20.0,
        samplerate: int = 11025,
        n_fft: int = 512,
        shifts: int = 4,
        pks_per_frame: int = 5,
        fanout: int = 3,
        freq_sd: float = 30.0,
        match_win: int = 1,
        min_count: int = 5,
        max_matches: int = 1,
        search_depth: int = 100,
        sort_by_time: bool = False,
        exact_count: bool = False,
        hashbits: int = 20,
        depth: int = 100,
        maxtime: int = 16384,
        verbose: bool = False,
        fail_on_error: bool = True,
        confident_match_threshold: int = 6,
    ):
        self._audfprint_analyze = audfprint_analyze
        self._audfprint_match = audfprint_match
        self._hash_table_module = hash_table

        # Setup the analyzer
        analyzer = audfprint_analyze.Analyzer(density=density)
        analyzer.target_sr = samplerate
        analyzer.n_fft = n_fft
        analyzer.n_hop = n_fft // 2
        analyzer.shifts = shifts
        analyzer.maxpksperframe = pks_per_frame
        analyzer.maxpairsperpeak = fanout
        analyzer.f_sd = freq_sd
        analyzer.fail_on_error = fail_on_error
        self.analyzer = analyzer

        # Setup the matcher
        matcher = audfprint_match.Matcher()
        matcher.window = match_win
        matcher.threshcount = min_count
        matcher.max_returns = max_matches
        matcher.search_depth = search_depth
        matcher.sort_by_time = sort_by_time
        matcher.exact_count = exact_count
        matcher.verbose = verbose
        self.matcher = matcher

        # Setup the hash table
        self.hash_table = hash_table.HashTable(
            hashbits=hashbits,
            depth=depth,
            maxtime=maxtime,
        )
        self.hash_table.params["samplerate"] = samplerate

        self.confident_match_threshold = confident_match_threshold

    @property
    def decision_rule(self) -> str:
        return f"accept if aligned_hashes >= {self.confident_match_threshold}"

    def train(self, file_list: list[str]) -> None:
        return None

    def add_tracks(self, track_files: list[str]) -> None:
        print(f"Indexing {len(track_files)} tracks with Shazam backend...")

        for file_path in tqdm(track_files, total=len(track_files)):
            hashes = self.analyzer.wavfile2hashes(file_path)
            track_id = os.path.basename(file_path)

            if len(hashes) == 0:
                print(f"Skipping {file_path}: no hashes extracted")
                continue

            self.hash_table.store(track_id, hashes)
            # try:
            #     hashes = self.analyzer.wavfile2hashes(file_path)
            #     track_id = os.path.basename(file_path)
            #
            #     if len(hashes) == 0:
            #         print(f"Skipping {file_path}: no hashes extracted")
            #         continue
            #
            #     self.hash_table.store(track_id, hashes)
            # except Exception as e:
            #     print(f"Skipping {file_path}: {e}")

    def search(self, query_path: str) -> MatchOutcome:
        query_hashes = self.analyzer.wavfile2hashes(query_path)

        if len(query_hashes) == 0:
            return MatchOutcome(
                predicted_track_id=None,
                score=0.0,
                score_name=self.score_name,
                higher_is_better=self.higher_is_better,
                metadata={"query_hash_count": 0},
            )

        results = self.matcher.match_hashes(self.hash_table, query_hashes)

        if results is None or len(results) == 0:
            return MatchOutcome(
                predicted_track_id=None,
                score=0.0,
                score_name=self.score_name,
                higher_is_better=self.higher_is_better,
                metadata={"query_hash_count": int(len(query_hashes))},
            )

        # audfprint result columns:
        # (id, filteredmatches, timoffs, rawmatches, origrank, mintime, maxtime)
        best = results[0]

        best_id_idx = int(best[0])
        filtered_matches = float(best[1])
        time_offset = int(best[2]) if len(best) > 2 else 0
        raw_matches = int(best[3]) if len(best) > 3 else 0
        # original_rank = int(best[4]) if len(best) > 4 else 0
        # min_time = int(best[5]) if len(best) > 5 else 0
        # max_time = int(best[6]) if len(best) > 6 else 0

        best_track_id = self.hash_table.names[best_id_idx]

        return MatchOutcome(
            predicted_track_id=best_track_id,
            score=filtered_matches,
            score_name=self.score_name,
            higher_is_better=self.higher_is_better,
            metadata={
                "query_hash_count": int(len(query_hashes)),
                "raw_matches": raw_matches,
                "time_offset_frames": time_offset,
                # "original_rank": original_rank,
                # "min_time_frame": min_time,
                # "max_time_frame": max_time,
            },
        )

    def is_confident_match(self, result: MatchOutcome) -> bool:
        if result.predicted_track_id is None:
            return False

        return result.score >= self.confident_match_threshold