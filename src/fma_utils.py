import os
import pandas as pd
import ast

def load_fma_csv(filepath):
    """
    Helper function (found in FMA example notebooks) to load FMA datasets with correct multi-index headers.
    """
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'),
                   ('artist', 'tags'), ('track', 'genres'),
                   ('track', 'genres_all')]

        for column in COLUMNS:
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

        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks
    return None

def attach_fma_paths(tracks_df, audio_root):
    """
    Generates the file path for each track in the FMA dataframe.
    Example: track_id 1234 -> .../001/001234.mp3
    """

    def get_path(track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(audio_root, tid_str[:3], tid_str + '.mp3')

    # Apply to the index (which contains track_id)
    return tracks_df.index.map(get_path)

def load_custom_paths(directory) -> pd.DataFrame:
    files = [f for f in os.listdir(directory) if f.endswith('.mp3') or f.endswith('.wav') or f.endswith('.flac')]
    df = pd.DataFrame(index=range(len(files)))  # Arbitrary index
    df['filepath'] = [os.path.join(directory, f) for f in files]
    df['track_title'] = files  # or parse filename
    return df

def create_dataset_df(fma_metadata_dir: str, fma_audio_files_dir, custom_audio_dir):
    print("--- Loading the Dataset ---")
    tracks = load_fma_csv(os.path.join(fma_metadata_dir, 'tracks.csv'))

    print("--- Linking Audio Files ---")
    tracks['filepath'] = attach_fma_paths(tracks, fma_audio_files_dir)
    valid_files_mask = [os.path.isfile(path) for path in tracks['filepath']]
    tracks = tracks[valid_files_mask]

    print(f"Tracks with valid audio files: {len(tracks)}")

    my_songs = load_custom_paths(custom_audio_dir)
    print(f"Custom songs found: {len(my_songs)}")

    return tracks, my_songs
