# app.py
import streamlit as st
# REMOVE: from transformers import pipeline # <-- Remove this top-level import
import torch # For st.cache_resource with type hinting if needed

# Page Configuration
st.set_page_config(
    page_title="Simple Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_sentiment_model():
    """Loads the sentiment analysis pipeline."""
    from transformers import pipeline # <--- IMPORT MOVED HERE
    print("Loading sentiment analysis model...")
    try:
        model = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model in Streamlit: {e}")
        return None

# Load the model
sentiment_pipeline = load_sentiment_model()

# ... rest of your app.py code remains the same ...

st.title("📝 Simple Sentiment Analyzer")
st.markdown("""
Enter some text below to analyze its sentiment (Positive or Negative).
This demo uses a pre-trained Transformer model (`distilbert-base-uncased-finetuned-sst-2-english`)
from Hugging Face.
""")

user_text = st.text_area("Enter text here:", "I love learning about AI and building cool projects!", height=150)

if st.button("Analyze Sentiment"):
    if sentiment_pipeline is None:
        st.error("Sentiment analysis model could not be loaded. Please check the logs or try again later.")
    elif not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Ensure pipeline is not None before using it
                if sentiment_pipeline:
                    results = sentiment_pipeline(user_text)
                    result = results[0]
                    label = result['label']
                    score = result['score']

                    st.subheader("Analysis Result:")
                    if label == "POSITIVE":
                        st.success(f"Sentiment: {label} (Confidence: {score:.4f})")
                        st.balloons()
                    elif label == "NEGATIVE":
                        st.error(f"Sentiment: {label} (Confidence: {score:.4f})")
                    else:
                        st.info(f"Sentiment: {label} (Confidence: {score:.4f})")
                else:
                    st.error("Sentiment analysis model is not available.")


            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                print(f"Error during analysis in Streamlit: {e}")

st.sidebar.info(
    "This app demonstrates sentiment analysis using a pre-trained Transformer model. "
    "No GPU is required for this demo."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Hugging Face 🤗 Transformers](https://huggingface.co/transformers) and [Streamlit](https://streamlit.io).")