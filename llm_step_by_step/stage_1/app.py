import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Word Similarity Finder", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    h1 {
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color: #1E3A8A;
        border-radius: 5px;
        border: 2px solid #1E3A8A;
        padding: 10px;
        font-size: 16px;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        border: none;
        padding: 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .dataframe {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .dataframe th {
        background-color: #1E3A8A;
        color: white;
        text-align: left;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 12px;
        border-top: 1px solid #f0f2f6;
    }
    
    .stAlert {
        background-color: #E5E7EB;
        color: #1E3A8A;
        border-radius: 5px;
        padding: 16px;
        margin-top: 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load GloVe Embeddings
@st.cache_data
def load_embeddings():
    embeddings_index = {}
    with open("glove.6B.100d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index


embeddings_index = load_embeddings()


def get_nearest_words(word, embeddings_index, top_n=10):
    if word not in embeddings_index:
        return None

    word_vector = embeddings_index[word]
    similarities = []

    for w, vec in embeddings_index.items():
        if w != word:
            similarity = 1 - cosine(word_vector, vec)
            similarities.append((w, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


st.title("Word Similarity Finder")

col1, col2 = st.columns([3, 1])

with col1:
    input_word = st.text_input(
        "Enter a word:", placeholder="e.g., computer, love, science"
    )

with col2:
    submit_button = st.button("Find Similar Words")

if submit_button and input_word:
    input_word = input_word.lower().strip()
    if input_word in embeddings_index:
        nearest_words = get_nearest_words(input_word, embeddings_index)

        df = pd.DataFrame(nearest_words, columns=["Word", "Cosine Similarity"])
        df["Cosine Similarity"] = df["Cosine Similarity"].apply(lambda x: f"{x:.4f}")

        st.success(f"Top 10 words most similar to '{input_word}':")
        st.table(df)
    else:
        st.error(
            f"'{input_word}' not found in the embeddings. Please try another word."
        )

# Add some information about the app
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 5px;'>
        <h3 style='color: #1E3A8A;'>About this app</h3>
        <p>This app uses GloVe word embeddings to find similar words based on cosine similarity.</p>
        <p>Enter a word and click 'Find Similar Words' to see the top 10 most similar words.</p>
    </div>
""",
    unsafe_allow_html=True,
)
