import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

def calculate_similarity(user_embeddings, book_embeddings, genre_embeddings, story_embeddings, book_df):
    pages, author_input, user_description_embedding, user_genre_embedding, user_story_embedding = user_embeddings

    if not np.allclose(user_description_embedding, np.zeros((1, user_description_embedding.shape[1]))):
        description_similarity = np.clip(cosine_similarity(user_description_embedding, book_embeddings).flatten(), 0,1)
    else:
        description_similarity = np.clip(np.ones(len(book_df)), 0,1)


    # Page length similarity
    if pages:
        pages = float(pages)
        book_df["pages"] = pd.to_numeric(book_df["pages"], errors="coerce")
        abs_diff = np.abs(book_df["pages"] - float(pages))
        norm_diff = abs_diff / pages * 2
        clipped = np.clip(norm_diff, 0, 1)
        page_length_similarity = np.clip(1 - clipped, 0,1)
    else:
        page_length_similarity = np.clip(np.ones(len(book_df)), 0,1)

    # Author similarity
    if author_input:
        author_similarity = np.clip(np.array([fuzz.partial_ratio(author_input, author) / 100 for author in book_df["author"]]), 0,1)
    else:
        author_similarity = np.clip(np.ones(len(book_df)), 0,1)

    # Genre similarity
    if not np.allclose(user_genre_embedding, np.zeros((1, user_genre_embedding.shape[1]))):
        genre_similarity = np.clip(cosine_similarity(user_genre_embedding, genre_embeddings).flatten(), 0,1)
    else:
        genre_similarity = np.clip(np.ones(len(book_df)), 0,1)

    # Story element similarity
    if not np.allclose(user_story_embedding, np.zeros((1, user_story_embedding.shape[1]))):
        story_element_similarity = np.clip(cosine_similarity(user_story_embedding, story_embeddings).flatten(), 0,1)
    else:
        story_element_similarity = np.clip(np.ones(len(book_df)), 0,1)

    with open("settings.json", "r") as f:
        weights:dict = json.load(f)  #{"description": 0.60,"author": 0.20,"genre": 0.15,"story_element": 0.00,"page_length": 0.05}

    # Combine similarities with weights
    combined_similarity = (
        (description_similarity * weights["description"]) +
        (author_similarity * weights["author"]) +
        (genre_similarity * weights["genre"]) +
        (story_element_similarity * weights["story_element"]) +
        (page_length_similarity * weights["page_length"])
    )

    return (
        combined_similarity,
        description_similarity,
        author_similarity,
        genre_similarity,
        story_element_similarity,
        page_length_similarity
    )
