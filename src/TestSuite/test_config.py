from dataclasses import dataclass

@dataclass
class TestConfig:
    """Configuration for a test."""
    f_s: int = 44100
    snippet_duration_sec: float = 5.0
    match_threshold: float = 0.3
    n_db_tracks: int = 100
    n_query_tracks: int = 20
    random_seed: int = 42
    save_passed_queries: bool = True
    save_failed_queries: bool = True
    temp_dir: str = "temp_queries"