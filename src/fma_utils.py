from pathlib import Path
import pandas as pd
import ast
from tqdm import tqdm


def load_fma_csv(filepath: str | Path) -> pd.DataFrame | None:
    """
    Helper function (found in FMA example notebooks) to load FMA datasets with correct multi-index headers.

    :arg filepath: The FMA csv file to load.
    :returns: FMA dataset as a ``pd.DataFrame``
    """
    filepath = Path(filepath)
    filename = filepath.name

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'),
                   ('artist', 'tags'), ('track', 'genres'),
                   ('track', 'genres_all')]

        for column in tqdm(COLUMNS):
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]

        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')

        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype('category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]

        for column in tqdm(COLUMNS):
            tracks[column] = tracks[column].astype('category')

        return tracks

    return None


def attach_fma_paths(tracks_df: pd.DataFrame, audio_root: str | Path):
    """
    Generates the file path for each track in the FMA dataframe.
    Example: track_id 1234 -> .../001/001234.mp3
    """
    audio_root = Path(audio_root)

    def get_path(track_id):
        tid_str = f"{track_id:06d}"
        return audio_root / tid_str[:3] / f"{tid_str}.mp3"

    # Apply to the index (which contains track_id)
    return tracks_df.index.map(get_path)


def load_custom_paths(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)

    files = []
    for file in tqdm(directory.iterdir(), desc="Processing custom song file extensions"):
        if file.is_file() and file.suffix.lower() in {'.mp3', '.wav', '.flac'}:
            files.append(file)

    df = pd.DataFrame(index=range(len(files))) # Arbitrary index
    df['filepath'] = files
    df["track_title"] = [file.name for file in files]

    return df


def create_dataset_df(
        fma_metadata_dir: str | Path,
        fma_audio_files_dir: str | Path,
        custom_audio_dir: str | Path
):
    tracks = load_fma_csv(Path(fma_metadata_dir) / "tracks.csv")

    tracks['filepath'] = attach_fma_paths(tracks, fma_audio_files_dir)
    valid_files_mask = tracks["filepath"].map(Path.is_file)
    tracks = tracks[valid_files_mask]

    print(f"Tracks with valid audio files: {len(tracks)}")

    my_songs = load_custom_paths(custom_audio_dir)
    print(f"Custom songs found: {len(my_songs)}")

    return tracks, my_songs
