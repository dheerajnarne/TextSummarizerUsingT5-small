import streamlit as st
from transformers import pipeline

# Initialize the text summarization pipeline with CPU
pipe = pipeline("text2text-generation", model="dheerajnarne/textsummarizer", device=-1)

# Set the page configuration for Streamlit
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "This is a text summarization app built with Streamlit and Hugging Face Transformers."
    },
)

# Apply dark theme using custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: #ffffff;
    }
    .stTextInput label, .stButton button {
        color: #ffffff;
    }
    .stButton button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("💬 Text Summarizer")
st.markdown(
    """
    Welcome to the **Text Summarizer**! Paste your text below, and our model will provide a concise summary.
    """
)

# Text input from the user
user_input = st.text_area(
    "Enter the text you want to summarize:",
    height=200,
    placeholder="Paste your text here...",
)

# Generate summary on button click
if st.button("Summarize Text"):
    if user_input.strip():
        # Call the summarization pipeline
        with st.spinner("Generating summary..."):
            summary = pipe(user_input, max_length=512, min_length=30, do_sample=False)[0]["generated_text"]
        
        # Display the summary
        st.markdown("### Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown(
    "Developed by **Narne Dheeraj Balaram** using [Hugging Face Transformers](https://huggingface.co/) and [Streamlit](https://streamlit.io/)."
)
