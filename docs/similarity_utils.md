# similarity_utils.py — Technical Function Explanations

## calculate_similarity

Calculates similarity scores between user preferences and books using multiple features and mathematical techniques.

```python
def calculate_similarity(user_embeddings, book_embeddings, genre_embeddings, story_embeddings, book_df):
    pages, author_input, user_description_embedding, user_genre_embedding, user_story_embedding = user_embeddings

    if not np.allclose(user_description_embedding, np.zeros((1, user_description_embedding.shape[1]))):
        description_similarity = np.clip(cosine_similarity(user_description_embedding, book_embeddings).flatten(), 0,1)
    else:
        description_similarity = np.clip(np.ones(len(book_df)), 0,1)

    # Page length similarity
    if pages:
        abs_diff = np.abs(book_df["pages"] - pages)
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

    weights: dict = {
        "description": 0.60,
        "author": 0.20,
        "genre": 0.15,
        "story_element": 0.00,
        "page_length": 0.05
    }

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
```

### Description, Genre, and Story Element Similarity

- **Cosine Similarity:**  
  For two vectors $\mathbf{a}$ and $\mathbf{b}$,  
  $$ \text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||} $$
  Since embeddings are normalized, this is simply their dot product.
- **Interpretation:**  
  Values range from -1 (opposite) to 1 (identical); here, all values are clipped to [0, 1] for scoring.

### Author Similarity

- **Fuzzy String Matching:**  
  Uses `fuzz.partial_ratio` to compare user input with each book's author (returns a score between 0 and 100, divided by 100 for normalization).
- **Purpose:**  
  Handles typos and partial matches, not just exact string equality.

### Page Length Similarity

- **Mathematical formula:**  
  $$ \text{abs\_diff} = |\text{book\_pages} - \text{user\_pages}| $$
  $$ \text{norm\_diff} = \frac{\text{abs\_diff}}{\text{user\_pages}} \times 2 $$
  $$ \text{clipped} = \min(\text{norm\_diff}, 1) $$
  $$ \text{page\_length\_similarity} = 1 - \text{clipped} $$
- **Interpretation:**  
  The closer the book's page count to the user's preference, the higher the similarity (max 1).

### Handling Missing User Input

- If a user input field (description, genre, or story element) is empty, a zero vector is used and the corresponding similarity is set to 1 for all books.
- If the author input is empty, author similarity is set to 1 for all books.
- If the page count is missing, page length similarity is set to 1 for all books.

### Combined Similarity

- **Weighted sum:**  
  Each feature's similarity is multiplied by a weight (sum to 1.0), reflecting its importance in the final score.
  - Description: 0.60
  - Author: 0.20
  - Genre: 0.15
  - Story element: 0.00
  - Page length: 0.05

### Output

- Returns a tuple of arrays:  
  (combined_similarity, description_similarity, author_similarity, genre_similarity, story_element_similarity, page_length_similarity)
