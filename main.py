from src.fma_utils import create_dataset_df

if __name__ == '__main__':
    METADATA_DIR = "B:\\Documents\\Uni\\MATH3092 Mathematics Project\\MusicDatasets\\fma\\fma_metadata"
    USER_SONGS_DIR = "B:\\Documents\\Uni\\MATH3092 Mathematics Project\\MusicDatasets\\my_songs"  # random extra songs that aren't part of the FMA dataset
    AUDIO_FILES_DIR = "B:\\Documents\\Uni\\MATH3092 Mathematics Project\\MusicDatasets\\fma\\fma_small\\fma_small"

    tracks, my_songs = create_dataset_df(METADATA_DIR, AUDIO_FILES_DIR, USER_SONGS_DIR)
