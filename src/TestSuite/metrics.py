from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .enums import TestType, TestStatus


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


class Top1Accuracy(Metric):
    name = "top1_accuracy_pct"
    label = "Top-1 Accuracy"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        pos = df[df["Test Type"] == TestType.POSITIVE]
        if pos.empty:
            return None
        return float((pos["Predicted ID"] == pos["Target ID"]).mean() * 100.0)


class IdentificationRate(Metric):
    name = "identification_rate_pct"
    label = "Identification Rate"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        pos = df[df["Test Type"] == TestType.POSITIVE]
        if pos.empty:
            return None
        return float((pos["Status"] == TestStatus.PASS).mean() * 100.0)


class TrueNegativeRate(Metric):
    name = "true_negative_rate_pct"
    label = "True Negative Rate"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        neg = df[df["Test Type"] == TestType.NEGATIVE]
        if neg.empty:
            return None
        return float((neg["Status"] == TestStatus.PASS).mean() * 100.0)


class AverageScoreCorrect(Metric):
    name = "avg_score_correct"
    label = "Average Score (Correct Prediction)"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
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
        neg = df[df["Test Type"] == TestType.NEGATIVE]
        if neg.empty:
            return None
        return float(neg["Score"].mean())


class AverageQueryTime(Metric):
    name = "avg_query_time_s"
    label = "Average Query Time"
    fmt = ".4f"

    def compute(self, df: pd.DataFrame) -> float | None:
        pos = df[df["Test Type"] == TestType.POSITIVE]
        if pos.empty:
            return None
        return float(pos["Query Time (s)"].mean())


DEFAULT_METRICS = [
    PositiveCount(),
    Top1Accuracy(),
    IdentificationRate(),
    AverageScoreCorrect(),
    AverageScoreWrong(),
    AverageQueryTime(),
    NegativeCount(),
    TrueNegativeRate(),
    AverageScoreNegative(),
]
