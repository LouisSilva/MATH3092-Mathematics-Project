from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .enums import TestType, TestStatus


def _exclude_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops trials that crashed before producing a real result.

    Errored trials carry placeholder values (predicted_track_id=None,
    is_confident=False, query_time_s=0) that would silently bias every
    metric: a crash on a positive query would look like a recall miss,
    a crash on a negative query would look like a true negative, and
    query_time_s=0 would pull the mean down. Crashes are reported
    separately via ``ErrorRate``.
    """
    return df[df["Status"] != TestStatus.ERROR]


class Metric(ABC):
    name: str
    label: str
    fmt: str = ""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> Any:
        """Calculate the metric from the test case DataFrame."""
        pass

    def format_value(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return "NaN"

        if isinstance(value, (float, int)) and self.fmt:
            return f"{value:{self.fmt}}"

        return str(value)


class PositiveCount(Metric):
    name = "n_positive"
    label = "Positive Queries"

    def compute(self, df: pd.DataFrame) -> int | None:
        val = (df["Test Type"] == TestType.POSITIVE).sum()
        return int(val) if val > 0 else None


class NegativeCount(Metric):
    name = "n_negative"
    label = "Negative Queries"

    def compute(self, df: pd.DataFrame) -> int | None:
        val = (df["Test Type"] == TestType.NEGATIVE).sum()
        return int(val) if val > 0 else None


class ErrorCount(Metric):
    """Number of trials that crashed before producing a result."""
    name = "n_errors"
    label = "Errored Queries"

    def compute(self, df: pd.DataFrame) -> int | None:
        val = (df["Status"] == TestStatus.ERROR).sum()
        return int(val) if val > 0 else None


class ErrorRate(Metric):
    """Percentage of trials that crashed before producing a result."""
    name = "error_rate_pct"
    label = "Error Rate (%)"
    fmt = ".2f"

    def compute(self, df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        return float((df["Status"] == TestStatus.ERROR).mean() * 100)


class AverageScoreCorrect(Metric):
    name = "avg_score_correct"
    label = "Average Score (Correct Prediction)"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        pos = df[df["Test Type"] == TestType.POSITIVE]
        correct = pos[pos["Predicted ID"] == pos["Target ID"]]
        if correct.empty:
            return None

        return float(correct["Score"].mean())


class AverageScoreWrong(Metric):
    name = "avg_score_wrong"
    label = "Average Score (Wrong Prediction)"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        pos = df[df["Test Type"] == TestType.POSITIVE]
        wrong = pos[pos["Predicted ID"] != pos["Target ID"]]
        if wrong.empty:
            return None
        return float(wrong["Score"].mean())


class AverageScoreNegative(Metric):
    name = "avg_score_negative"
    label = "Average Negative Score"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        neg = df[df["Test Type"] == TestType.NEGATIVE]
        if neg.empty:
            return None
        return float(neg["Score"].mean())


class RawTop1Accuracy(Metric):
    """Threshold-independent accuracy. It measures if the correct track was the first result, regardless of confidence."""
    name = "raw_top1_accuracy_pct"
    label = "Raw Top-1 Accuracy (%)"
    fmt = ".2f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        pos = df[df["Test Type"] == TestType.POSITIVE]
        if pos.empty:
            return None
        return float((pos["Predicted ID"] == pos["Target ID"]).mean() * 100)


class TruePositiveRate(Metric):
    """Measures the percentage of positive queries that were correctly identified and passed the confidence threshold. It is also called recall."""
    name = "recall_pct"
    label = "Recall (%)"
    fmt = ".2f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        pos = df[df["Test Type"] == TestType.POSITIVE]
        if pos.empty:
            return None

        is_true_positive = (pos["Predicted ID"] == pos["Target ID"]) & pos["Is Confident"]
        return float(is_true_positive.mean() * 100)


class FalsePositiveRate(Metric):
    """Measures the percentage of negative (alien) queries that falsely passed the confidence threshold. It is also called the false alarm rate."""
    name = "false_pos_rate_pct"
    label = "False Positive Rate (%)"
    fmt = ".2f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)
        neg = df[df["Test Type"] == TestType.NEGATIVE]
        if neg.empty:
            return None

        return float(neg["Is Confident"].mean() * 100)


class Precision(Metric):
    """Measures the percentage of confident matches that are actually correct."""
    name = "precision_pct"
    label = "Precision (%)"
    fmt = ".2f"

    def compute(self, df: pd.DataFrame) -> float | None:
        df = _exclude_errors(df)

        # True positives: confident match AND correct ID
        true_positives = ((df["Test Type"] == TestType.POSITIVE) &
                          df["Is Confident"] &
                          (df["Predicted ID"] == df["Target ID"])).sum()

        # False positives: confident match on an alien track
        false_positives_alien = ((df["Test Type"] == TestType.NEGATIVE) & df["Is Confident"]).sum()

        # False positives: confident match on a DB track, but wrong ID
        false_positives = ((df["Test Type"] == TestType.POSITIVE) &
                           df["Is Confident"] &
                           (df["Predicted ID"] != df["Target ID"])).sum()

        total_false_positives = false_positives_alien + false_positives

        if (true_positives + total_false_positives) == 0:
            return None

        return float(true_positives / (true_positives + total_false_positives) * 100)


class MeanQueryTime(Metric):
    name = "mean_query_time_s"
    label = "Mean Query Time"
    fmt = ".3f"

    def compute(self, df: pd.DataFrame) -> float | None:
        if df.empty:
            return None

        return float(df["Query Time (s)"].mean())


DEFAULT_METRICS = [
    RawTop1Accuracy(),
    TruePositiveRate(),
    FalsePositiveRate(),
    Precision(),
    MeanQueryTime(),
    ErrorRate(),
]
