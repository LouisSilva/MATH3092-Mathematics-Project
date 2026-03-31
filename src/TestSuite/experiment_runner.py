import math
import os
import re
import secrets
import string
import librosa
import shutil
import time
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from typing import Any

from .test_cases import AudioTestCase
from .test_config import TestConfig
from .test_status import TestStatus
from .test_type import TestType

from ..spectrogram_generation import load_audio
from ..fingerprint_database import FingerprintDatabase

def random_suffix(length=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def sanitize_filename(text: str) -> str:
    """Removes special characters from a string to make it safe for filenames."""
    return re.sub(r'[^\w\-. ]', '_', text)

class ExperimentRunner:
    def __init__(self, fingerprinter, search_strategy, all_filepaths: list[str], config: TestConfig):
        if not all_filepaths:
            raise ValueError("File path list cannot be empty.")

        self.fingerprinter = fingerprinter
        self.search_strategy = search_strategy
        self.all_filepaths = all_filepaths
        self.config = config
        self.db: FingerprintDatabase | None = None
        self.db_tracks: list[str] = []
        self.query_tracks: list[str] = []
        self.alien_tracks: list[str] = []
        self.results: list[dict[str, Any]] = []
        self.rng = np.random.default_rng(self.config.random_seed)

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

    def setup_database(self):
        """Selects tracks for DB and queries, then populates the database."""
        print("--- Setting up database ---")
        n_total = len(self.all_filepaths)
        n_db = self.config.n_db_tracks
        n_query = self.config.n_query_tracks

        if n_db + (n_query * 2) > n_total:
            # We need enough for DB + Positive Queries + Negative Queries (Aliens)
            raise ValueError(f"Not enough tracks! Need {n_db + 2 * n_query}, but have {n_total}.")

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

        print(f"Selected {len(self.db_tracks)} tracks for database.")
        print(f"Selected {len(self.query_tracks)} tracks for positive queries.")
        print(f"Selected {len(self.alien_tracks)} tracks for negative/alien queries.")

        self.db = FingerprintDatabase(self.fingerprinter, self.search_strategy)
        self.db.add_tracks(self.db_tracks)
        print("--- Database setup complete ---")

    def _run_single_query(self, query_path: str, test_case: AudioTestCase, test_type: TestType) -> dict[str, Any]:
        """Processes a single audio file query."""
        if self.db is None:
            raise RuntimeError("Database not set up. Call setup_database() first.")

        target_id = os.path.basename(query_path)
        result_entry = {
            "Test Case": str(test_case),
            "Test Type": test_type,
            "Target ID": target_id,
            "Predicted ID": None,
            "Distance": float('inf'),
            "Query Time (s)": 0.0,
            "Status": TestStatus.ERROR
        }

        temp_filename = f"{test_type}_{sanitize_filename(str(test_case))}_{sanitize_filename(target_id)}.wav"
        temp_file_path = os.path.join(self.config.temp_dir, temp_filename)

        try:
            # Load audio and select a random snippet
            full_duration = librosa.get_duration(path=query_path)
            max_offset = max(0, math.floor(full_duration - self.config.snippet_duration_sec))
            offset = self.rng.uniform(0, max_offset)

            audio_snippet, sr = load_audio(
                query_path,
                f_s=self.config.f_s,
                offset=offset,
                duration=self.config.snippet_duration_sec
            )

            # Apply some function to the audio (e.g. add white noise, distortion, reverb, etc.)
            distorted_audio = test_case.apply(audio_snippet, sr)

            # Write to a temp file
            sf.write(temp_file_path, distorted_audio, sr)

            # Run the search algorithm and record how much time it takes
            start_time = time.perf_counter()
            predicted_id, dist = self.db.search(temp_file_path)
            end_time = time.perf_counter()

            # Record the results
            result_entry["Predicted ID"] = predicted_id
            result_entry["Distance"] = dist
            result_entry["Query Time (s)"] = end_time - start_time

            # Process the different test types
            if test_type == TestType.POSITIVE:
                is_match = (predicted_id == target_id)
                is_confident = (dist <= self.config.match_threshold)
                result_entry["Status"] = TestStatus.PASS if (is_match and is_confident) else TestStatus.FAIL

            elif test_type == TestType.NEGATIVE:
                result_entry["Status"] = TestStatus.PASS if dist > self.config.match_threshold else TestStatus.FAIL

        except Exception as e:
            print(f"Error processing {query_path}: {e}")
            result_entry["Status"] = TestStatus.ERROR

        finally:
            # Cleanup the temp files
            if os.path.exists(temp_file_path):
                status = result_entry["Status"]

                # Move if config enabled for that status, otherwise delete
                if status == TestStatus.PASS and self.config.save_passed_queries:
                    shutil.move(temp_file_path, os.path.join(self.passed_dir, temp_filename))

                elif (status == TestStatus.FAIL or status == TestStatus.ERROR) and self.config.save_failed_queries:
                    # We only care about the positive test types
                    if test_type == TestType.POSITIVE:
                        shutil.move(temp_file_path, os.path.join(self.failed_dir, temp_filename))

                else:
                    os.remove(temp_file_path)

        return result_entry

    def run_experiment(self, test_cases: list[AudioTestCase]):
        """Runs the evaluation across all specified test cases."""
        if not self.db:
            raise RuntimeError("Database not set up. Call setup_database() first.")

        for case in test_cases:
            print(f"\n--- Running Test Case: {case} ---")

            # Positive Control: Query with tracks that should be in the DB
            positive_tasks = [(path, case, TestType.POSITIVE) for path in self.query_tracks]

            # Negative Control: Query with tracks that are not in the DB
            negative_tasks = [(path, case, TestType.NEGATIVE) for path in self.alien_tracks]

            all_tasks = positive_tasks + negative_tasks
            for task in tqdm(all_tasks, desc=f"Querying ({str(case)})"):
                self.results.append(self._run_single_query(*task))

    def get_results_df(self) -> pd.DataFrame:
        """Returns the collected results as a pandas DataFrame."""
        if not self.results: return pd.DataFrame()
        return pd.DataFrame(self.results)

    def summarize_results(self, results_df: pd.DataFrame):
        """Prints a detailed summary of the experiment results."""
        if results_df.empty:
            print("No results to summarize.")
            return

        print("\n--- Experiment Summary ---")

        for test_case_name, group in results_df.groupby('Test Case'):
            print(f"\n--- Results for: {test_case_name} ---")

            pos_group = group[group['Test Type'] == TestType.POSITIVE]
            neg_group = group[group['Test Type'] == TestType.NEGATIVE]

            # Positive Control Metrics
            if not pos_group.empty:
                # Raw Identification Accuracy (Ignoring Threshold)
                raw_hits = (pos_group['Predicted ID'] == pos_group['Target ID'])
                raw_accuracy = raw_hits.mean() * 100

                # Verified Accuracy (Using Threshold)
                verified_accuracy = (pos_group['Status'] == TestStatus.PASS).mean() * 100

                avg_time = pos_group['Query Time (s)'].mean()

                # Distances for Correct IDs vs Incorrect IDs
                correct_matches = pos_group[raw_hits]
                incorrect_matches = pos_group[~raw_hits]

                avg_match_dist = correct_matches['Distance'].mean() if not correct_matches.empty else float('nan')
                avg_mismatch_dist = incorrect_matches['Distance'].mean() if not incorrect_matches.empty else float(
                    'nan')

                print(f"Positive Queries: {len(pos_group)}")
                print(f"Top-1 Accuracy: {raw_accuracy:.2f}% (Found correct ID)")
                print(f"Verified Pass Rate: {verified_accuracy:.2f}% (Correct ID + Dist < Threshold)")
                print(f"Avg Distance (Match): {avg_match_dist:.4f}")
                print(f"Avg Distance (Wrong): {avg_mismatch_dist:.4f}")
                print(f"Avg Query Time: {avg_time:.4f}s")

            # Negative Control Metrics
            if not neg_group.empty:
                true_negative_rate = (neg_group['Status'] == TestStatus.PASS).mean() * 100
                avg_rejection_dist = neg_group['Distance'].mean()

                print(f"Negative Queries: {len(neg_group)}")
                print(f"True Negative Rate: {true_negative_rate:.2f}% (Correctly Rejected)")
                print(f"Avg Rejection Dist: {avg_rejection_dist:.4f}")