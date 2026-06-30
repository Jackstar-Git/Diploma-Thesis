# embedding_utils.py — Technical Function Explanations

## extract_embeddings

Encodes a string or list of strings into high-dimensional vector embeddings using a pre-trained SentenceTransformer model.

```python
def extract_embeddings(text: str | list[str]) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True, show_progress_bar=True, batch_size=1024)
```

- **Mathematical aspect:**  
  Each input string is mapped to a vector in ℝⁿ (n = embedding dimension, e.g., 384 or 768).  
  If `normalize_embeddings=True`, each vector is L2-normalized:  
  $$ \mathbf{v}_{norm} = \frac{\mathbf{v}}{||\mathbf{v}||_2} $$
- **Batching:**  
  Processes up to 1024 texts at once for efficiency.
- **Output:**  
  Returns a numpy array of shape (num_texts, embedding_dim).

---

## get_cached_embeddings

Loads or generates and caches embeddings for book descriptions, genres, and story elements.

```python
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
```

- **Purpose:**  
  Avoids recomputing embeddings by saving them to disk (serialization with pickle).
- **If** cache files are missing or empty, computes embeddings and saves them.
- **Returns:**  
  Tuple of numpy arrays for descriptions, genres, and story elements.

---

## create_user_embeddings

Creates embeddings for user input fields, handling empty strings by returning zero vectors of the correct shape.

```python
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
```

- **Technical note:**  
  Each user input string is embedded and reshaped to (1, embedding_dim) for compatibility with pairwise similarity functions.  
  If a field is empty, a zero vector of the correct shape is used as a placeholder.
- **Returns:**  
  Tuple containing the original page count, author input, and the three embeddings.
