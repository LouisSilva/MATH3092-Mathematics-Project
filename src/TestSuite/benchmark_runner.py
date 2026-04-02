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
from .metrics import Metric, DEFAULT_METRICS
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
            backends: list[RetrievalBackend],
            config: BenchmarkConfig,
            all_filepaths: list[str],
    ):
        if not backends:
            raise ValueError("You must provide at least one backend.")

        if not all_filepaths:
            raise ValueError("File path list cannot be empty.")

        self.db_tracks: list[str] = []
        self.query_tracks: list[str] = []
        self.alien_tracks: list[str] = []
        self.query_offsets: dict[str, float] = {}

        self.config: BenchmarkConfig = config
        self.backends: list[RetrievalBackend] = backends
        self.results: dict[str, BenchmarkReport] = {backend.backend_name: BenchmarkReport() for backend in self.backends}

        self.all_filepaths: list[str] = all_filepaths # TODO: Put this in the BenchmarkConfig object
        self.rng: np.random.Generator = np.random.default_rng(self.config.random_seed) # TODO: Maybe this too?

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
        print("Setting up database...")
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

        for backend in self.backends:
            backend.add_tracks(self.db_tracks)

        print("Database setup complete")

    def train(self, training_filepaths: list[str]) -> None:
        print(f"Training backends on {len(training_filepaths)} files.")
        for backend in self.backends:
            backend.train(training_filepaths)

    def _run_single_query(
            self,
            query_path: str,
            test_case: AudioTestCase,
            test_type: TestType
    ) -> dict[str, BenchmarkTrial]:
        """
        Processes one audio file query across all backends.

        :arg query_path:
        :arg test_case:
        :arg test_type:
        :returns: A dictionary of ``BenchmarkTrial`` objects indexed by the name of the backend the benchmark was performed on.
        """
        target_track_id = Path(query_path).stem
        temp_query_filename = f"{test_type}_{sanitize_filename(str(test_case))}_{sanitize_filename(target_track_id)}.wav"
        temp_query_filepath = os.path.join(self.config.temp_dir, temp_query_filename)

        trials = {}

        try:
            # Load the audio and select a random snippet
            audio_snippet, sr = load_audio(
                query_path,
                f_s=self.config.f_s,
                offset=self.query_offsets[query_path],
                duration=self.config.snippet_duration_sec
            )

            # Apply some transformation to the audio (e.g. add white noise, distortion, reverb, etc.)
            transformed_audio = test_case.apply(audio_snippet, sr, self.rng)

            # Write it to a temp file
            sf.write(temp_query_filepath, transformed_audio, sr)

            for backend in self.backends:
                # Initialize the MatchOutcome and BenchmarkTrial objects
                match_outcome = MatchOutcome(
                    predicted_track_id=None,
                    score=backend.initial_score,
                    score_name=backend.score_name,
                    higher_is_better=backend.higher_is_better,
                    metadata={}
                )

                benchmark_trial = BenchmarkTrial(
                    test_case=str(test_case),
                    test_type=test_type,
                    target_track_id=target_track_id,
                    match_outcome=match_outcome,
                    query_time_s=0,
                    status=TestStatus.ERROR
                )

                try:
                    # Run the search algorithm and record how much time it takes
                    start_time = time.perf_counter()
                    match_outcome: MatchOutcome = backend.search(temp_query_filepath)
                    end_time = time.perf_counter()

                    # Record the results
                    benchmark_trial.query_time_s = end_time - start_time
                    benchmark_trial.match_outcome = match_outcome

                    is_confident = backend.is_confident_match(match_outcome) # TODO: Add this to the BenchmarkTrial object possibly

                    # Process the different test types
                    if test_type == TestType.POSITIVE:
                        is_match = match_outcome.predicted_track_id == target_track_id
                        benchmark_trial.status = TestStatus.PASS if (is_match and is_confident) else TestStatus.FAIL

                    elif test_type == TestType.NEGATIVE:
                        benchmark_trial.status = TestStatus.PASS if not is_confident else TestStatus.FAIL

                except Exception as e:
                    print(f"Error processing {query_path} with {backend.backend_name}: {e}")

                trials[backend.backend_name] = benchmark_trial

        except Exception as e:
            print(f"Error generating transformed audio for {query_path}: {e}")

            # If the audio generation fails, then the trial for all backends fail
            for backend in self.backends:
                trials[backend.backend_name] = BenchmarkTrial(
                    test_case=str(test_case),
                    test_type=test_type,
                    target_track_id=target_track_id,
                    match_outcome=MatchOutcome(None, backend.initial_score, backend.score_name, backend.higher_is_better),
                    query_time_s=0,
                    status=TestStatus.ERROR,
                )

        finally:
            # Cleanup the temp files
            if os.path.exists(temp_query_filepath):
                any_failed = any(trial.status in (TestStatus.FAIL, TestStatus.ERROR) for trial in trials.values())

                # If any backend FAILED, and saving FAILED queries is enabled in the config, then the query audio shall be saved
                if any_failed and self.config.save_failed_queries and test_type == TestType.POSITIVE:
                    shutil.move(temp_query_filepath, os.path.join(self.failed_dir, temp_query_filename))

                # If they DIDN'T FAIL, and saving PASSED queries is enabled in the config, then the query audio shall be saved
                elif not any_failed and self.config.save_passed_queries and test_type == TestType.POSITIVE:
                    shutil.move(temp_query_filepath, os.path.join(self.passed_dir, temp_query_filename))

                # Otherwise, delete the query audio
                else:
                    os.remove(temp_query_filepath)

        return trials

    def run_experiment(self, test_cases: list[AudioTestCase]) -> None:
        """Runs the specified test cases across all backends."""
        for case in test_cases:
            print(f"\n--- Running Test Case: {case} ---")

            # Positive Control: Query with tracks that should be in the DB
            positive_tasks = [(path, case, TestType.POSITIVE) for path in self.query_tracks]

            # Negative Control: Query with tracks that are not in the DB
            negative_tasks = [(path, case, TestType.NEGATIVE) for path in self.alien_tracks]

            all_tasks = positive_tasks + negative_tasks
            for task in tqdm(all_tasks, desc=f"Querying ({str(case)})"):
                trials_dict = self._run_single_query(*task)

                for backend_name, trial in trials_dict.items():
                    self.results[backend_name].add(trial)

    def get_results_df(self) -> pd.DataFrame:
        """Returns the collected results as a single pandas ``DataFrame`` with a backend column."""
        dfs = []
        for backend_name, report in self.results.items():
            df = report.to_dataframe()

            if not df.empty:
                df.insert(0, "Backend", backend_name)
                dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def summarize_results(self, metrics: list[Metric] | None = None) -> None:
        """Prints a detailed summary of the experiment results for all backends."""
        if metrics is None:
            metrics = DEFAULT_METRICS

        for backend in self.backends:
            print(f"\n{'-' * 20}")
            print(f"BACKEND: {backend.backend_name}")
            print(f"{'-' * 20}")

            self.results[backend.backend_name].print_summary(
                score_name=backend.score_name,
                higher_is_better=backend.higher_is_better,
                decision_rule=backend.decision_rule,
                metrics=metrics
            )
