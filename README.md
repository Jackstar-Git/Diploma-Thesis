# Book Matcher — Diploma Thesis

---

## 1. Table of Contents

* [2. About the Project](#2-about-the-project)
* [3. Features](#3-features)
* [4. How it Works (Technical Methodology & Algorithms)](#4-how-it-works-technical-methodology--algorithms)
* [5. Built With (Technologies)](#5-built-with-technologies)
* [6. Requirements (System Requirements & Libraries)](#6-requirements-system-requirements--libraries)
* [7. Usage](#7-usage)
* [8. Repository Structure](#8-repository-structure)
* [9. License & Copyright](#9-license--copyright)
* [10. Author](#10-author)

---

## 2. About the Project

The **Book Matcher** is a prototype content-based recommendation system (*Content-Based Filtering*). It was developed as a functional prototype for a high school diploma thesis (HAK-Diplomarbeit) to empirically investigate the impact of AI-driven product recommendations on the human decision-making process. The thesis uses **Thalia Buch und Medien GmbH** as an industrial example for the use-case study.

> [!NOTE]
> All brand names, logos, or specific references to "Thalia Buch und Medien GmbH" used within this project or documentation remain the intellectual property of their respective owners.

The primary goal of the system is to counteract information overload and increase the visibility of niche products. Unlike collaborative filtering, this approach is robust against the **"Cold-Start" problem**. The system uses a structured dataset of the **50,000 best-selling books** to test theoretical sorting algorithms in practice through a *mixed-methods approach* and the *Starfish model*.

---

## 3. Features

* **Multilingual Text Vectorization:** Transforms inputs into numerical vectors across 15 global languages.
* **Semantic Feature Space:** Utilizes `model/all-Minilm-L12-v2` for precise 384-dimensional mapping.
* **Dynamic Weighting:** Users can adjust attribute importance (description, author, genre, story, pages) in real-time.
* **Performant Caching:** Uses `pickle` to store vectors, ensuring subsequent queries are near-instant.
* **Interactive UI:** Dynamic result table with expandable descriptions.

---

## 4. How it Works (Technical Methodology & Algorithms)

### Semantic Similarity

For text-based attributes, we use the **cosine similarity** formula. To ensure GitHub compatibility, we denote the math as follows:

$$cosine\\_sim(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \cdot \|\mathbf{b}\|_2}$$

### Page Length Deviation

The similarity $S$ for page counts is calculated via:

$$S_{pages} = 1 - \min\left(1, \frac{|book\\_pages - user\\_pages|}{user\\_pages} \cdot 2\right)$$

### Total Weighted Score

The final ranking is a weighted sum:

$$S = w_{desc} \cdot S_{desc} + w_{author} \cdot S_{author} + w_{genre} \cdot S_{genre} + w_{story} \cdot S_{story} + w_{pages} \cdot S_{pages}$$

> [!TIP]
> The default weights are optimized as: Description (45%), Author (20%), Genre (15%), Story Elements (15%), and Page Length (5%).

---

## 5. Built With (Technologies)

* **Backend:** Python, SentenceTransformers, Pandas, NumPy, Scikit-Learn, FuzzyWuzzy.
* **Frontend:** Flask, Jinja2.

---

## 6. Requirements (System Requirements & Libraries)

* **Python:** 3.12+
* **Dependencies:** `sentence-transformers`, `pandas`, `numpy`, `scikit-learn`, `fuzzywuzzy`, `flask`.

> [!WARNING]
> The `model/` folder is empty in this repo due to size constraints. The model will auto-download (approx. 90MB) on first execution. Ensure at least 4GB of RAM is available.

---
## 7. Usage

1. Run `main.py` to start the local server.
> [!NOTE]
> Use the `--no-gui` flag to only start the engine without starting the frontend and flask-server
2. Navigate to `http://127.0.0.1:5000/`.
3. Use the search bar for book discovery or the `/settings` page to customize weighting.

---
## 8. Repository Structure

```text
├── data/
│   ├── books_small.csv          # smaller dataset for testing
│   └── books.csv                # 50,000 book main dataset
├── docs/
│   ├── embedding_utils.md
│   ├── main.md
│   └── similarity_utils.md
├── embeddings/
│   ├── additional_embeddings.pkl # Genre & Story vectors
│   └── book_embeddings.pkl       # Description vectors
├── gui/
│   ├── static/
│   │   ├── about.css
│   │   ├── home.css
│   │   ├── home.js
│   │   ├── Logo.png
│   │   ├── navbar.css
│   │   └── style.css            # root styling
│   └── templates/
│       ├── about.jinja-html
│       ├── index.jinja-html     # Main search UI
│       ├── info.jinja-html
│       ├── navbar.jinja-html
│       └── settings.jinja-html  # Weight configuration UI
├── model/                       # Download destination for AI models - empty due to storage reasons
├── modules/
│   ├── embedding_utils.py
│   ├── main.py
│   └── similarity_utils.py
├── static/
├── .gitignore                   # Excludes venv, pycache, and model files
├── .gitattributes
├── app.py
├── main.py                      # Application entry point
├── req.txt.txt                  # Project dependencies
└── settings.json                # The settings for the weights
```

---

## 9. License & Copyright

**Copyright (c) March 2026 by Jackstar-Git**

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

* **Copyleft:** Any modifications or derivative works must also be licensed under GPL-3.0.
* **Third-Party:** All references to "Thalia Buch und Medien GmbH" are used for educational purposes within the scope of a diploma thesis and remain the property of their respective owners.

---

## 10. Author
Developed by **Jackstar-Git** (Jackstar) for HAK-Diplomarbeit 2026.
