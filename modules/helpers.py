import time
from modules.embedding_utils import get_cached_embeddings, create_user_embeddings
from modules.similarity_utils import calculate_similarity
import numpy as np
import pandas as pd



# Load the book dataset
book_df: pd.DataFrame = pd.read_csv("data/books.csv", quoting=2).fillna("")
book_df["pages"] = pd.to_numeric(book_df["pages"], errors="coerce")

# Cache files for embeddings
embedding_cache_file: str = "embeddings/book_embeddings.pkl"
additional_cache_file: str = "embeddings/additional_embeddings.pkl"

def print_recommended_books(sorted_by="author_similarity", num_books=5, acented=False, num_places=3):
    recommended_books = book_df.sort_values(by=sorted_by, ascending=acented).head(num_books)
    print(f"Recommended Books and Their {sorted_by.replace('_', ' ').title()} Scores:")

    for _, row in recommended_books.iterrows():
        print(f"\nBook: {row['title']} by {row['author']}")
        print(f"  Page Length: {row['pages']} pages")
        print(f"  Genres: {row["genres"]}")
        print(f"  Story Elements: {row['story_elements']}")
        #print(f"  Description: {row['description']}")
        print(f"\nSimilarity Scores:")
        print(f"  Description Similarity: {row['description_similarity'] * 100:.{num_places}f}%")
        print(f"  Author Similarity: {row['author_similarity'] * 100:.{num_places}f}%")
        print(f"  Genre Similarity: {row['genre_similarity'] * 100:.{num_places}f}%")
        print(f"  Story Element Similarity: {row['story_element_similarity'] * 100:.{num_places}f}%")
        print(f"  Page Length Similarity: {row['page_length_similarity'] * 100:.{num_places}f}%")
        print(f"  Combined Similarity: {row['similarity'] * 100:.{num_places}f}%")
        print("-" * 50)


def fetch_books_api(api_input: tuple, num_of_results: int = 5) -> list[dict]:
    embeddings_tuple = get_cached_embeddings(book_df, embedding_cache_file, additional_cache_file)
    book_embeddings: np.ndarray = embeddings_tuple[0]
    genre_embeddings: np.ndarray = embeddings_tuple[1]
    story_embeddings: np.ndarray = embeddings_tuple[2]

    user_embeddings_tuple = create_user_embeddings(api_input)
    pages: int | None = user_embeddings_tuple[0]
    author_input: str = user_embeddings_tuple[1]
    user_description_embedding: np.ndarray = user_embeddings_tuple[2]
    user_genre_embedding: np.ndarray = user_embeddings_tuple[3]
    user_story_embedding: np.ndarray = user_embeddings_tuple[4]
    user_embeddings = (pages, author_input, user_description_embedding, user_genre_embedding, user_story_embedding)

    similarity_tuple = calculate_similarity(
        user_embeddings, book_embeddings, genre_embeddings, story_embeddings, book_df
    )
    similarity: np.ndarray = similarity_tuple[0]
    description_similarity: np.ndarray = similarity_tuple[1]
    author_similarity: np.ndarray = similarity_tuple[2]
    genre_similarity: np.ndarray = similarity_tuple[3]
    story_element_similarity: np.ndarray = similarity_tuple[4]
    page_length_similarity: np.ndarray = similarity_tuple[5]

    book_df["similarity"] = similarity
    book_df["description_similarity"] = description_similarity
    book_df["author_similarity"] = author_similarity
    book_df["genre_similarity"] = genre_similarity
    book_df["story_element_similarity"] = story_element_similarity
    book_df["page_length_similarity"] = page_length_similarity

    return book_df.sort_values(by="similarity", ascending=False).head(num_of_results).to_dict(orient="records")
