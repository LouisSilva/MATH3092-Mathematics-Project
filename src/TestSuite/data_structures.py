import pandas as pd
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any


class TestType(Enum):
    POSITIVE = "Positive"  # Query is in the DB
    NEGATIVE = "Negative"  # Query is NOT in the DB (Alien)

    def __str__(self):
        return self.value


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"

    def __str__(self):
        return self.value


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

    @property
    def is_correct_id(self) -> bool:
        return self.match_outcome.predicted_track_id == self.target_track_id

    @property
    def is_pass(self) -> bool:
        return self.status == TestStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """
        Flat dict for DataFrame export.
        Metadata is expanded into prefixed columns.
        """
        base = {
            "Test Case": self.test_case,
            "Test Type": self.test_type,
            "Target ID": self.target_track_id,
            "Predicted ID": self.match_outcome.predicted_track_id,
            "Score": self.match_outcome.score,
            "Score Name": self.match_outcome.score_name,
            "Higher Is Better": self.match_outcome.higher_is_better,
            "Query Time (s)": self.query_time_s,
            "Status": self.status,
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
    n_positive: int
    n_negative: int
    top1_accuracy_pct: float | None
    verified_pass_rate_pct: float | None
    true_negative_rate_pct: float | None
    avg_score_correct: float | None
    avg_score_wrong: float | None
    avg_score_negative: float | None
    avg_query_time_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    f_s: int = 44100
    snippet_duration_sec: float = 5.0
    n_db_tracks: int = 100
    n_query_tracks: int = 20
    random_seed: int = 42
    save_passed_queries: bool = True
    save_failed_queries: bool = True
    temp_dir: str = "temp_queries"


@dataclass
class BenchmarkReport:
    """
    A collection of ``BenchmarkTrial`` objects plus summary/export helpers.
    """
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

    def summarize(self) -> list[TestCaseReport]:
        if not self.benchmark_trials:
            return []

        df = self.to_dataframe()
        summaries: list[TestCaseReport] = []

        for test_case_name, group in df.groupby("Test Case"):
            pos_group = group[group["Test Type"] == TestType.POSITIVE]
            neg_group = group[group["Test Type"] == TestType.NEGATIVE]

            top1_accuracy_pct: float | None = None
            verified_pass_rate_pct: float | None = None
            true_negative_rate_pct: float | None = None
            avg_score_correct: float | None = None
            avg_score_wrong: float | None = None
            avg_score_negative: float | None = None
            avg_query_time_s: float | None = None

            if not pos_group.empty:
                raw_hits = pos_group["Correct ID"]
                top1_accuracy_pct = float(raw_hits.mean() * 100.0)
                verified_pass_rate_pct = float((pos_group["Status"] == TestStatus.PASS).mean() * 100.0)
                avg_query_time_s = float(pos_group["Query Time (s)"].mean())

                correct_matches = pos_group[raw_hits]
                incorrect_matches = pos_group[~raw_hits]

                if not correct_matches.empty:
                    avg_score_correct = float(correct_matches["Score"].mean())
                if not incorrect_matches.empty:
                    avg_score_wrong = float(incorrect_matches["Score"].mean())

            if not neg_group.empty:
                true_negative_rate_pct = float((neg_group["Status"] == TestStatus.PASS).mean() * 100.0)
                avg_score_negative = float(neg_group["Score"].mean())

            summaries.append(
                TestCaseReport(
                    test_case=str(test_case_name),
                    n_positive=int(len(pos_group)),
                    n_negative=int(len(neg_group)),
                    top1_accuracy_pct=top1_accuracy_pct,
                    verified_pass_rate_pct=verified_pass_rate_pct,
                    true_negative_rate_pct=true_negative_rate_pct,
                    avg_score_correct=avg_score_correct,
                    avg_score_wrong=avg_score_wrong,
                    avg_score_negative=avg_score_negative,
                    avg_query_time_s=avg_query_time_s,
                )
            )

        return summaries

    def print_summary(self, *, score_name: str, higher_is_better: bool, decision_rule: str) -> None:
        if not self.benchmark_trials:
            print("No results to summarize.")
            return

        print("\n--- Experiment Summary ---")
        print(f"Metric: {score_name}")
        print(f"Higher is better: {higher_is_better}")
        print(f"Decision rule: {decision_rule}")

        for summary in self.summarize():
            print(f"\n--- Results for: {summary.test_case} ---")

            if summary.n_positive > 0:
                print(f"Positive Queries: {summary.n_positive}")
                print(f"Top-1 Accuracy: {summary.top1_accuracy_pct:.2f}%")
                print(f"Verified Pass Rate: {summary.verified_pass_rate_pct:.2f}%")

                if summary.avg_score_correct is not None:
                    print(f"Avg Score (Correct ID): {summary.avg_score_correct:.4f}")
                else:
                    print("Avg Score (Correct ID): nan")

                if summary.avg_score_wrong is not None:
                    print(f"Avg Score (Wrong ID): {summary.avg_score_wrong:.4f}")
                else:
                    print("Avg Score (Wrong ID): nan")

                if summary.avg_query_time_s is not None:
                    print(f"Avg Query Time: {summary.avg_query_time_s:.4f}s")
                else:
                    print("Avg Query Time: nan")

            if summary.n_negative > 0:
                print(f"Negative Queries: {summary.n_negative}")
                print(f"True Negative Rate: {summary.true_negative_rate_pct:.2f}%")

                if summary.avg_score_negative is not None:
                    print(f"Avg Negative Score: {summary.avg_score_negative:.4f}")
                else:
                    print("Avg Negative Score: nan")
