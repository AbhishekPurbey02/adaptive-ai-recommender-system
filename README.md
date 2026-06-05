# Adaptive AI Recommender System

An end-to-end machine learning recommendation system that suggests relevant e-commerce products using popularity-based, content-based, and hybrid recommendation strategies. The project is built around Amazon product data and deployed as an interactive Streamlit web app where users can search for a product, tune recommendation weights and compare different recommendation approaches.

> Live Demo: https://adaptive-ai-recommender-system-lab.streamlit.app/

## Project Overview

Modern e-commerce platforms depend on recommendation engines to improve product discovery, personalization, and user engagement. This project simulates that workflow by building a recommendation system that ranks products using product metadata, customer purchase signals and text similarity.

The system starts with basic popularity ranking, adds content-based recommendations using TF-IDF and cosine similarity, and then combines both signals in a hybrid model. The Streamlit app makes the model usable through a simple search interface and adjustable hybrid weights.

## Key Features

- Popularity-based recommender using recent purchase demand
- Content-based recommender using product title, category, price segment, and bestseller signals
- Hybrid recommendation engine combining content similarity and popularity score
- Adjustable content and popularity weights from the Streamlit sidebar
- Product search with top-N recommendation control
- Cached data and model loading for faster Streamlit inference
- Notebook-driven development workflow for preprocessing, modeling, and experimentation

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Pickle

## Dataset

The project uses a cleaned Amazon product dataset with 1,000 product records. Key fields include:

- Product title
- Product URL and image URL
- Rating stars
- Price and list price
- Category ID
- Bestseller flag
- Bought in last month count

These features are used to build product text representations, estimate popularity, and generate recommendations.

## Recommendation Approaches

### 1. Popularity-Based Recommendation

Ranks products by normalized purchase demand using the `boughtInLastMonth` field. This works well for showing trending or high-demand products even when the user has not provided detailed preference information.

### 2. Content-Based Recommendation

Creates a combined product text feature from title, category, price range, and bestseller status. TF-IDF converts this text into vectors, and cosine similarity finds products most similar to the searched item.

### 3. Hybrid Recommendation

Combines content similarity and popularity into a single ranking score:

```text
hybrid_score = (content_weight * content_similarity) + (popularity_weight * popularity_score)
```

This balances relevance with real-world demand, making recommendations more useful than relying on only one signal.

## Project Structure

```text
adaptive-ai-recommender-system/
|-- app/
|   |-- streamlit_app.py
|-- artifacts/
|   |-- nn_model.pkl
|   |-- tfidf_vectorizer.pkl
|-- data/
|   |-- amazon_products.csv
|   |-- amazon_small.csv
|-- notebooks/
|   |-- data_preprocessing.ipynb
|   |-- popularity_recommender.ipynb
|   |-- content_based_recommender.ipynb
|   |-- hybrid_recommender.ipynb
|-- requirements.txt
|-- README.md
```

## How to Run Locally

1. Clone the repository:

```bash
git clone <repository-url>
cd adaptive-ai-recommender-system
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

4. Search for a product such as:

```text
Luggage
Socks
Shirt
Suitcase
Shoes
```

## App Workflow

1. Load Amazon product data.
2. Create text features from product metadata.
3. Vectorize product text using TF-IDF.
4. Calculate normalized popularity scores.
5. Generate recommendations using the selected model.
6. Display top-N results with price, rating, demand, similarity, or hybrid score.

## Project Highlights

- Built an adaptive e-commerce recommendation system using Python, Scikit-learn, and Streamlit.
- Implemented popularity-based, content-based, and hybrid recommendation models using TF-IDF, cosine similarity, and purchase-demand signals.
- Designed an interactive Streamlit interface with searchable recommendations, adjustable hybrid weights, and top-N result control.
- Processed Amazon product metadata including ratings, prices, categories, bestseller flags, and recent purchase counts.
- Deployed the machine learning application through Streamlit for live portfolio access.

## Future Improvements

- Add collaborative filtering using user-item interaction data
- Include product images in the recommendation results
- Store user feedback and adapt rankings over time
- Add offline evaluation metrics such as precision@k and recall@k
- Improve deployment configuration and app performance for larger datasets

## Author

Developed as a practical machine learning project to demonstrate recommendation system design, feature engineering, model deployment, and interactive ML app development.
