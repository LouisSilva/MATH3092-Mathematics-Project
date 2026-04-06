from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .enums import TestStatus, TestType
from .metrics import Metric


@dataclass(slots=True)
class MatchOutcome:
    """
    A search result that doesn't depend on the specific backend used.

    :ivar predicted_track_id: The predicted best matching track, or ``None`` if no confident hit exists.
    :ivar score: The backend-specific match score.
    :ivar score_name: Human-readable name of the score metric.
    :ivar higher_is_better: Whether larger scores mean better matches.
    :ivar metadata: Optional backend-specific details.
    """
    predicted_track_id: str | None
    score: float
    score_name: str
    higher_is_better: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkTrial:
    """
    The result for one query under one test case.
    """
    test_case: str
    test_type: TestType
    target_track_id: str
    match_outcome: MatchOutcome
    query_time_s: float
    status: TestStatus
    is_confident: bool = False

    @property
    def is_correct_id(self) -> bool:
        return self.match_outcome.predicted_track_id == self.target_track_id

    @property
    def is_pass(self) -> bool:
        return self.status == TestStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary for DataFrame export. Metadata is expanded into prefixed columns."""
        base = {
            "Test Case": self.test_case,
            "Test Type": self.test_type,
            "Target ID": self.target_track_id,
            "Predicted ID": self.match_outcome.predicted_track_id,
            "Score": self.match_outcome.score,
            "Score Name": self.match_outcome.score_name, # TODO: remove this
            "Higher Is Better": self.match_outcome.higher_is_better, # TODO: remove this
            "Query Time (s)": self.query_time_s,
            "Status": self.status,
            "Is Confident": self.is_confident
        }

        for key, value in self.match_outcome.metadata.items():
            base[f"Meta: {key}"] = value

        return base


@dataclass(slots=True)
class TestCaseReport:
    """
    The results for a single test case.
    """
    test_case: str
    metrics: list[Metric]
    results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"Test Case": self.test_case, **self.results}

    def to_lines(self) -> list[str]:
        lines = [f"\n--- Results for: {self.test_case} ---"]

        for metric in self.metrics:
            value = self.results.get(metric.name)

            if value is None or pd.isna(value): continue
            lines.append(f"{metric.label}: {metric.format_value(value)}")

        return lines


@dataclass
class BenchmarkConfig:  # TODO: Write docstring with all arguments
    f_s: int = 44100
    snippet_duration_sec: float = 5
    num_db_tracks: int = 100
    num_positive_queries: int = 50
    num_negative_queries: int = 10
    random_seed: int = 42
    save_passed_queries: bool = True
    save_failed_queries: bool = True
    temp_dir: str = "temp_queries"

    def __post_init__(self):
        if self.num_positive_queries > self.num_db_tracks:
            raise ValueError(
                f"Cannot have more positive queries ({self.num_positive_queries}) than tracks in the database ({self.num_db_tracks}). Positive queries must be a subset of the database.")


@dataclass
class BenchmarkReport:
    """A collection of ``BenchmarkTrial`` objects plus summary/export helpers."""
    benchmark_trials: list[BenchmarkTrial] = field(default_factory=list)

    def add(self, result: BenchmarkTrial) -> None:
        self.benchmark_trials.append(result)

    def extend(self, results: list[BenchmarkTrial]) -> None:
        self.benchmark_trials.extend(results)

    def __len__(self) -> int:
        return len(self.benchmark_trials)

    def is_empty(self) -> bool:
        return len(self.benchmark_trials) == 0

    def to_dataframe(self) -> pd.DataFrame:
        if not self.benchmark_trials:
            return pd.DataFrame()
        return pd.DataFrame([item.to_dict() for item in self.benchmark_trials])

    def summarize(self, metrics: list[Metric]) -> list[TestCaseReport]:
        if not self.benchmark_trials:
            return []

        df = self.to_dataframe()
        summaries: list[TestCaseReport] = []

        for test_case_name, group in df.groupby("Test Case"):
            results = {metric.name: metric.compute(group) for metric in metrics}

            summaries.append(
                TestCaseReport(
                    test_case=str(test_case_name),
                    metrics=metrics,
                    results=results
                )
            )

        return summaries

    def print_summary(
            self,
            *,
            score_name: str,
            higher_is_better: bool,
            decision_rule: str,
            metrics: list[Metric]
    ) -> None:
        if not self.benchmark_trials:
            print("No results to summarize.")
            return

        print("\n--- Experiment Summary ---")
        print(f"Metric: {score_name}")
        print(f"Higher is better: {higher_is_better}")
        print(f"Decision rule: {decision_rule}")

        for summary in self.summarize(metrics):
            print("\n".join(summary.to_lines()))


def format_float(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.4f}"
