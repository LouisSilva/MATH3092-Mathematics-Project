import os
import re
import secrets
import shutil
import string
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .data_structures import TestType, TestStatus, MatchOutcome, BenchmarkTrial, BenchmarkConfig, BenchmarkReport
from .test_cases import AudioTestCase
from ..EigenSpectraFingerprinter.spectrogram_generation import load_audio
from ..retrieval_backends import RetrievalBackend


def random_suffix(length=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def sanitize_filename(text: str) -> str:
    """Removes special characters from a string to make it safe for filenames."""
    return re.sub(r'[^\w\-. ]', '_', text)


class BenchmarkRunner:
    def __init__(
            self,
            backend: RetrievalBackend,
            config: BenchmarkConfig,
            all_filepaths: list[str],
    ):
        if not all_filepaths:
            raise ValueError("File path list cannot be empty.")

        self.db_tracks: list[str] = []
        self.query_tracks: list[str] = []
        self.alien_tracks: list[str] = []
        self.query_offsets: dict[str, float] = {}

        self.backend: RetrievalBackend = backend
        self.config: BenchmarkConfig = config
        self.all_filepaths: list[str] = all_filepaths
        self.results: BenchmarkReport = BenchmarkReport()
        self.rng: np.random.Generator = np.random.default_rng(self.config.random_seed)

        # Ensure base temp dir exists
        os.makedirs(self.config.temp_dir, exist_ok=True)

        # Setup directory structure
        self.passed_dir = os.path.join(self.config.temp_dir, "passed")
        self.failed_dir = os.path.join(self.config.temp_dir, "failed")

        # Clean temp directories if saving is enabled
        if self.config.save_passed_queries:
            os.makedirs(self.passed_dir, exist_ok=True)
        if self.config.save_failed_queries:
            os.makedirs(self.failed_dir, exist_ok=True)

    def setup_database(self) -> None:
        """Selects tracks for DB and queries, then populates the database."""
        print("--- Setting up database ---")
        n_total = len(self.all_filepaths)
        n_db = self.config.n_db_tracks
        n_query = self.config.n_query_tracks

        # if n_db + (n_query * 2) > n_total:
        if n_db + n_query > n_total:
            # We need enough for DB + Positive Queries + Negative Queries (Aliens)
            raise ValueError(f"Not enough tracks! Need {n_db + n_query}, but have {n_total}.")

        # Shuffle all file paths reproducibly
        shuffled_indices = self.rng.permutation(n_total)

        # Select DB Tracks
        db_indices = shuffled_indices[:n_db]
        self.db_tracks = [self.all_filepaths[i] for i in db_indices]

        # Select Positive Queries (Subset of DB)
        query_indices_subset = self.rng.choice(db_indices, size=n_query, replace=False)
        self.query_tracks = [self.all_filepaths[i] for i in query_indices_subset]

        # Select Alien/Negative Queries (Tracks NOT in DB)
        alien_indices = shuffled_indices[n_db: n_db + n_query]
        self.alien_tracks = [self.all_filepaths[i] for i in alien_indices]

        for path in self.query_tracks + self.alien_tracks:
            full_duration = librosa.get_duration(path=path)
            max_offset = max(0.0, full_duration - self.config.snippet_duration_sec)
            self.query_offsets[path] = float(self.rng.uniform(0.0, max_offset))

        print(f"Selected {len(self.db_tracks)} tracks for database.")
        print(f"Selected {len(self.query_tracks)} tracks for positive queries.")
        print(f"Selected {len(self.alien_tracks)} tracks for negative/alien queries.")

        self.backend.add_tracks(self.db_tracks)
        print("Database setup complete")

    def train(self, training_filepaths: list[str]) -> None:
        print(f"Training backend on {len(training_filepaths)} files.")
        self.backend.train(training_filepaths)

    def _initial_score_value(self) -> float:
        return 0.0 if self.backend.higher_is_better else float("inf")

    def _run_single_query(
            self,
            query_path: str,
            test_case: AudioTestCase,
            test_type: TestType
    ) -> BenchmarkTrial:
        """Processes one audio file query."""
        target_id = Path(query_path).stem

        match_outcome = MatchOutcome(
            predicted_track_id=None,
            score=self._initial_score_value(),
            score_name=self.backend.score_name,
            higher_is_better=self.backend.higher_is_better,
            metadata={},
        )

        benchmark_trial = BenchmarkTrial(
            test_case=str(test_case),
            test_type=test_type,
            target_track_id=target_id,
            match_outcome=match_outcome,
            query_time_s=0.0,
            status=TestStatus.ERROR,
        )

        temp_filename = f"{test_type}_{sanitize_filename(str(test_case))}_{sanitize_filename(target_id)}.wav"
        temp_file_path = os.path.join(self.config.temp_dir, temp_filename)

        try:
            # Load audio and select a random snippet

            audio_snippet, sr = load_audio(
                query_path,
                f_s=self.config.f_s,
                offset=self.query_offsets[query_path],
                duration=self.config.snippet_duration_sec
            )

            # Apply some function to the audio (e.g. add white noise, distortion, reverb, etc.)
            distorted_audio = test_case.apply(audio_snippet, sr, self.rng)

            # Write to a temp file
            sf.write(temp_file_path, distorted_audio, sr)

            # Run the search algorithm and record how much time it takes
            start_time = time.perf_counter()
            match_outcome: MatchOutcome = self.backend.search(temp_file_path)
            end_time = time.perf_counter()

            # Record the results
            benchmark_trial.query_time_s = end_time - start_time
            benchmark_trial.match_outcome = match_outcome

            is_confident = self.backend.is_confident_match(match_outcome)

            # Process the different test types
            if test_type == TestType.POSITIVE:
                is_match = match_outcome.predicted_track_id == target_id
                benchmark_trial.status = TestStatus.PASS if (is_match and is_confident) else TestStatus.FAIL

            elif test_type == TestType.NEGATIVE:
                benchmark_trial.status = TestStatus.PASS if not is_confident else TestStatus.FAIL

        except Exception as e:
            print(f"Error processing {query_path}: {e}")
            benchmark_trial.status = TestStatus.ERROR

        finally:
            # Cleanup the temp files
            if os.path.exists(temp_file_path):

                # Move if config enabled for that status, otherwise delete
                if benchmark_trial.status == TestStatus.PASS and self.config.save_passed_queries:
                    shutil.move(temp_file_path, os.path.join(self.passed_dir, temp_filename))

                elif (
                        benchmark_trial.status == TestStatus.FAIL or benchmark_trial.status == TestStatus.ERROR) and self.config.save_failed_queries:
                    # We only care about the positive test types
                    if test_type == TestType.POSITIVE:
                        shutil.move(temp_file_path, os.path.join(self.failed_dir, temp_filename))

                else:
                    os.remove(temp_file_path)

        return benchmark_trial

    def run_experiment(self, test_cases: list[AudioTestCase]) -> None:
        """Runs all the specified test cases."""
        for case in test_cases:
            print(f"\n--- Running Test Case: {case} ---")

            # Positive Control: Query with tracks that should be in the DB
            positive_tasks = [(path, case, TestType.POSITIVE) for path in self.query_tracks]

            # Negative Control: Query with tracks that are not in the DB
            negative_tasks = [(path, case, TestType.NEGATIVE) for path in self.alien_tracks]

            all_tasks = positive_tasks + negative_tasks
            for task in tqdm(all_tasks, desc=f"Querying ({str(case)})"):
                self.results.add(self._run_single_query(*task))

    def get_results_df(self) -> pd.DataFrame:
        """Returns the collected results as a pandas DataFrame."""
        return self.results.to_dataframe()

    def summarize_results(self) -> None:
        """Prints a detailed summary of the experiment results."""
        self.results.print_summary(
            score_name=self.backend.score_name,
            higher_is_better=self.backend.higher_is_better,
            decision_rule=self.backend.decision_rule,
        )
