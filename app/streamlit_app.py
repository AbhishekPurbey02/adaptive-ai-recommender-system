# ============================================
# ADAPTIVE AI RECOMMENDER SYSTEM - WEB APP
# Day 6: Streamlit Frontend
# ============================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Adaptive AI Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD DATA & MODELS (Cached for performance)
# ============================================

@st.cache_resource
def load_data():
    """Load product data"""
    df = pd.read_csv("data/amazon_small.csv")
    return df

@st.cache_resource
def load_vectorizer():
    """Load TF-IDF vectorizer"""
    try:
        with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Loaded existing vectorizer")
    except:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        print("✅ Created new vectorizer")
    return vectorizer

@st.cache_resource
def prepare_data(df, _vectorizer):
    """Create TF-IDF matrix and calculate popularity scores"""
    
    # Create product features
    def create_text(row):
        features = [str(row['title']), f"category_{row['category_id']}"]
        if row['price'] > 0:
            if row['price'] < 50:
                features.append("budget")
            elif row['price'] < 200:
                features.append("mid_range")
            else:
                features.append("premium")
        if row['isBestSeller']:
            features.append("bestseller")
        return " ".join(features)
    
    df['product_text'] = df.apply(create_text, axis=1)
    
    # Create TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df['product_text'])
    
    # Calculate popularity scores
    max_bought = df['boughtInLastMonth'].max()
    df['popularity_score'] = df['boughtInLastMonth'] / max_bought if max_bought > 0 else 0
    
    return tfidf_matrix, df

# Load everything
df = load_data()
vectorizer = load_vectorizer()
tfidf_matrix, df = prepare_data(df, vectorizer)

# ============================================
# RECOMMENDATION FUNCTIONS
# ============================================

def recommend_popular(n=5):
    """Popularity-based recommendations"""
    return df.nlargest(n, 'popularity_score')[['title', 'price', 'stars', 'boughtInLastMonth']]

def recommend_content(product_name, n=5):
    """Content-based recommendations"""
    matching = df[df['title'].str.contains(product_name, case=False)]
    if len(matching) == 0:
        return None
    
    product_idx = matching.index[0]
    content_scores = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix)[0]
    similar_indices = content_scores.argsort()[::-1][1:n+1]
    
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            'title': df.iloc[idx]['title'],
            'price': df.iloc[idx]['price'],
            'stars': df.iloc[idx]['stars'],
            'similarity': round(content_scores[idx], 3)
        })
    
    return pd.DataFrame(recommendations)

def recommend_hybrid(product_name, n=5, content_weight=0.7, popularity_weight=0.3):
    """Hybrid recommendations (Content + Popularity)"""
    matching = df[df['title'].str.contains(product_name, case=False)]
    if len(matching) == 0:
        return None
    
    product_idx = matching.index[0]
    content_scores = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix)[0]
    
    recommendations = []
    for idx in range(len(df)):
        if idx != product_idx:
            content_score = content_scores[idx]
            popularity_score = df.iloc[idx]['popularity_score']
            hybrid_score = (content_weight * content_score) + (popularity_weight * popularity_score)
            
            recommendations.append({
                'title': df.iloc[idx]['title'],
                'price': df.iloc[idx]['price'],
                'stars': df.iloc[idx]['stars'],
                'bought': df.iloc[idx]['boughtInLastMonth'],
                'hybrid_score': round(hybrid_score, 3)
            })
    
    recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return pd.DataFrame(recommendations[:n])

# ============================================
# UI - SIDEBAR
# ============================================

st.sidebar.title(" Adaptive AI Recommender")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["🔥 Hybrid (Recommended)", "📊 Popularity", "🎯 Content-Based"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Hybrid Weights")
content_weight = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.7, 0.1)
popularity_weight = st.sidebar.slider("Popularity Weight", 0.0, 1.0, 0.3, 0.1)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **How it works:**
    - 📊 Popularity: Trending products
    - 🎯 Content_based: Similar products  
    - 🔥 Hybrid: Best of both

    """
)

# ============================================
# UI - MAIN PAGE
# ============================================

st.title(" Adaptive AI Recommender System")
st.markdown("*AI-powered E-commerce Recommendation Engine*")

st.markdown("---")

# Search section
col1, col2 = st.columns([3, 1])
with col1:
    search_term = st.text_input(
        "🔍 **Search for a product**",
        placeholder="e.g., Luggage, Socks, Shirt, Suitcase, Shoes...",
        help="Enter any product name to get recommendations"
    )

with col2:
    num_recommendations = st.number_input(
        " **Number of recommendations**",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

st.markdown("---")

# ============================================
# GENERATE RECOMMENDATIONS
# ============================================

if search_term:
    st.subheader(f" Recommendations for: **{search_term}**")
    st.markdown("---")
    
    if model_choice == "📊 Popularity":
        with st.spinner("Finding popular products..."):
            results = recommend_popular(num_recommendations)
        
        if results is not None and len(results) > 0:
            st.success(f"✅ Found {len(results)} popular recommendations")
            
            for i, row in results.iterrows():
                with st.container():
                    cols = st.columns([5, 1, 1, 1])
                    with cols[0]:
                        st.write(f"**{i+1}. {row['title']}**")
                    with cols[1]:
                        st.write(f"💰 ${row['price']}")
                    with cols[2]:
                        st.write(f"⭐ {row['stars']}")
                    with cols[3]:
                        st.write(f"📦 {row['boughtInLastMonth']} bought")
                    st.divider()
        else:
            st.warning("No products found. Try a different search term.")
    
    elif model_choice == "🎯 Content-Based":
        with st.spinner("Finding similar products..."):
            results = recommend_content(search_term, num_recommendations)
        
        if results is not None and len(results) > 0:
            st.success(f"✅ Found {len(results)} similar products")
            
            for i, row in results.iterrows():
                with st.container():
                    cols = st.columns([5, 1, 1, 1])
                    with cols[0]:
                        st.write(f"**{i+1}. {row['title']}**")
                    with cols[1]:
                        st.write(f"💰 ${row['price']}")
                    with cols[2]:
                        st.write(f"⭐ {row['stars']}")
                    with cols[3]:
                        st.write(f"🔗 {row['similarity']:.0%} match")
                    st.divider()
        else:
            st.warning(f"Product '{search_term}' not found. Try a different search term.")
    
    else:  # Hybrid model
        with st.spinner("Generating hybrid recommendations..."):
            results = recommend_hybrid(
                search_term, 
                num_recommendations, 
                content_weight, 
                popularity_weight
            )
        
        if results is not None and len(results) > 0:
            st.success(f"✅ Found {len(results)} hybrid recommendations")
            st.caption(f"⚙️ **Weights:** {content_weight:.0%} Content + {popularity_weight:.0%} Popularity")
            
            for i, row in results.iterrows():
                with st.container():
                    cols = st.columns([5, 1, 1, 1])
                    with cols[0]:
                        st.write(f"**{i+1}. {row['title']}**")
                    with cols[1]:
                        st.write(f"💰 ${row['price']}")
                    with cols[2]:
                        st.write(f"⭐ {row['stars']}")
                    with cols[3]:
                        st.write(f"🎯 Score: {row['hybrid_score']}")
                    st.divider()
        else:
            st.warning(f"Product '{search_term}' not found. Try a different search term.")

else:
    # Show sample when no search
    st.info("🔍 **Enter a product name above to get recommendations!**")
    
    st.subheader(" 🔥 Most Popular Products Right Now")
    sample = recommend_popular(5)
    
    for i, row in sample.iterrows():
        st.write(f"**{i+1}. {row['title'][:80]}...**")
        st.write(f"   💰 ${row['price']} | ⭐ {row['stars']} | 📦 {row['boughtInLastMonth']} bought last month")
        st.divider()

# # ============================================
# # FOOTER
# # ============================================

# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center'>
#         <p>Built with ❤️ using Streamlit | Adaptive AI Recommender System</p>
#         <p style='font-size: 12px; color: gray;'>Day 6: Web Application | Day 7: Coming Soon</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )