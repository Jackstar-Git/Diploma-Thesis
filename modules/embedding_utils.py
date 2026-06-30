import pickle
from typing import Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

model_path: str = r"model/all-Minilm-L12-v2" #fast model
#model_path: str = r"model/paraphrase-multilingual-MiniLM-L12-v2" #multilingual model

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model: SentenceTransformer = SentenceTransformer(model_path, local_files_only=True).to(device)

def extract_embeddings(text: str | list[str]) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True, show_progress_bar=True, batch_size=1024)

def get_cached_embeddings(book_df: pd.DataFrame, embedding_cache_file: str, additional_cache_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with open(embedding_cache_file, "rb") as f:
            book_embeddings: np.ndarray = pickle.load(f)
    except (FileNotFoundError, EOFError):
        book_embeddings = extract_embeddings(book_df["description"].tolist())
        with open(embedding_cache_file, "wb") as f:
            pickle.dump(book_embeddings, f)

    try:
        with open(additional_cache_file, "rb") as f:
            genre_embeddings, story_embeddings = pickle.load(f)
    except (FileNotFoundError, EOFError):
        genre_embeddings = extract_embeddings(book_df["genres"].tolist())
        story_embeddings = extract_embeddings(book_df["story_elements"].tolist())
        with open(additional_cache_file, "wb") as f:
            pickle.dump((genre_embeddings, story_embeddings), f)

    return book_embeddings, genre_embeddings, story_embeddings

def create_user_embeddings(user_input: tuple[int | None, str, str, str, str]) -> tuple[int | None, str, np.ndarray, np.ndarray, np.ndarray, tuple]:
    pages, author_input, genres_input, story_elements_input, description_input = user_input

    # Embed user input, with deterministic handling for empty strings
    if description_input.strip() == "":
        print("No description provided, using default embedding.")
        user_description_embedding: np.ndarray = np.zeros((1, model.get_sentence_embedding_dimension()), dtype=np.float32)
    else:
        user_description_embedding: np.ndarray = extract_embeddings(description_input).reshape(1, -1)

    if genres_input.strip() == "":
        print("No genres provided, using default embedding.")
        user_genre_embedding: np.ndarray = np.zeros((1, model.get_sentence_embedding_dimension()), dtype=np.float32)
    else:
        user_genre_embedding: np.ndarray = extract_embeddings(genres_input).reshape(1, -1)

    if story_elements_input.strip() == "":
        print("No story elements provided, using default embedding.")
        user_story_embedding: np.ndarray = np.zeros((1, model.get_sentence_embedding_dimension()), dtype=np.float32)
    else:
        user_story_embedding: np.ndarray = extract_embeddings(story_elements_input).reshape(1, -1)

    return (pages, author_input, user_description_embedding, user_genre_embedding, user_story_embedding)

