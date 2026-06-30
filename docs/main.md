# main.py — Technical and Mathematical Explanations

## Overview

This script implements a full recommendation pipeline:
1. Loads a book dataset.
2. Generates or loads vector embeddings for books and user input.
3. Computes similarity scores between user preferences and all books using multiple features.
4. Ranks and prints the top recommended books, including a breakdown of similarity components.

---

## Data Loading

```python
book_df: pd.DataFrame = pd.read_csv("data/books.csv", quoting=2).fillna("")
book_df["pages"] = pd.to_numeric(book_df["pages"], errors="coerce")
```

- **Technical note:**  
  Loads the dataset and ensures the "pages" column is numeric for mathematical operations.

---

## Embedding Caching

```python
embedding_cache_file: str = "embeddings/book_embeddings.pkl"
additional_cache_file: str = "embeddings/additional_embeddings.pkl"
```

- **Purpose:**  
  Avoids recomputation by storing precomputed embeddings on disk.

---

## print_recommended_books

Prints the top recommended books and their similarity scores.

```python
def print_recommended_books(sorted_by="similarity", num_books=5, acented=False):
    recommended_books = book_df.sort_values(by=sorted_by, ascending=acented).head(num_books)
    # ...print details and similarity breakdown...
```

- **Technical note:**  
  Sorts the DataFrame by the chosen similarity metric and prints the top results.

---

## Main Pipeline

```python
if __name__ == "__main__":
    user_input = (300, "Suzan Colins", "", "", "thrilling fight for survivel in a broken soceity")
    # ...load/generate embeddings...
    # ...create user embeddings...
    # ...calculate similarities...
    # ...update DataFrame...
    # ...print recommendations and timing...
```

### Steps:

1. **Embedding Generation:**  
   Uses `get_cached_embeddings` to obtain vector representations for all books, genres, and story elements.
2. **User Embedding:**  
   Converts user input into the same vector space for direct comparison.
3. **Similarity Calculation:**  
   Uses `calculate_similarity` to compute a weighted sum of feature similarities (see `similarity_utils.md` for math).
4. **Ranking:**  
   Sorts books by combined similarity score and prints the top matches.

### Mathematical Details

- **Cosine similarity** is used for vector-based features (description, genre, story elements).
- **Fuzzy string matching** is used for author names.
- **Page length similarity** is based on normalized absolute difference.
- **Combined similarity** is a weighted sum:
  $$ S = w_1 \cdot S_{genre} + w_2 \cdot S_{author} + w_3 \cdot S_{story} + w_4 \cdot S_{desc} + w_5 \cdot S_{pages} $$
  where $w_i$ are the weights (sum to 1).

---

## Timing

- The script prints the time taken for each major step (embedding, user embedding, similarity calculation), useful for profiling and optimization.

---

## Key Points

- Efficient use of vectorization and caching.
- Modular design for easy extension.
- Mathematical rigor in similarity computation and ranking.
