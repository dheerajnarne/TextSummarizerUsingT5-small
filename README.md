# Text Summarizer using T5 Transformer

This repository contains a Jupyter Notebook for building and fine-tuning a text summarization model using the **T5 Transformer**. The project leverages **Hugging Face's Transformers library** for model handling and **Hugging Face Datasets** for training and evaluation. Additionally, a **Streamlit Application** is provided for an interactive summarization experience.

Text summarization is a key task in Natural Language Processing (NLP), and this implementation provides a scalable approach to train a summarization model tailored to custom datasets.

---

## Overview

### **Model: T5 Transformer**
The **T5 (Text-to-Text Transfer Transformer)** model by Google treats all NLP tasks as a text-to-text problem. This allows a unified approach to tasks like summarization, translation, and classification. In this project:
- The **T5-Small** version is used as a base model.
- It is fine-tuned for summarization tasks using custom datasets.

### **Dataset**
The dataset used for this project is the **`antash420/text-summarization-alpaca-format`**, accessible from Hugging Face's dataset hub. This dataset contains text data formatted specifically for summarization tasks. The data is split into:
- **Training Set**: 2,000 samples.
- **Validation Set**: 300 samples.
- **Test Set**: 100 samples.

### **Objective**
The primary goal is to fine-tune the T5 model to summarize input texts efficiently while maintaining relevance and coherence.

---

## Installation

Before running the notebook or the Streamlit application, ensure you have installed the required dependencies:

```bash
pip install -U transformers datasets tensorboard sentencepiece accelerate evaluate rouge_score streamlit
```

These libraries are critical for:
- Loading pre-trained models (`transformers`).
- Accessing datasets (`datasets`).
- Monitoring training progress (`tensorboard`).
- Evaluating summarization performance (`evaluate`, `rouge_score`).
- Building the Streamlit application (`streamlit`).

---

## Workflow

### **Notebook Workflow**

1. **Install Dependencies**: All necessary libraries are installed directly within the notebook.
2. **Load the Pre-trained Model**:
    - The tokenizer and T5 base model (`google-t5/t5-small`) are loaded using Hugging Face's API.
3. **Dataset Preprocessing**:
    - The dataset is downloaded and split into training, validation, and testing subsets.
    - Samples are shuffled and prepared for input-output tokenization.
4. **Fine-tuning**:
    - The model is fine-tuned using the `Trainer` API, which simplifies training workflows with built-in features like logging and evaluation.
    - Training configurations include batch size, learning rate, and the number of epochs.
5. **Evaluation**:
    - The model's performance is measured using the ROUGE metric.
    - Generates summaries for test samples and compares them with reference outputs.

### **Streamlit Application Workflow**

1. **Interactive Summarization**:
    - The Streamlit app provides a user-friendly interface to input text and generate summaries interactively.
    - Users can paste or type text, click the "Summarize Text" button, and view the generated summary in real-time.
2. **Dark Theme and Custom Styling**:
    - The app is styled with a dark theme for a modern look and feel.
    - Proper spacing is provided to display summaries clearly and concisely.
3. **Backend**:
    - The app uses the fine-tuned T5 model hosted on Hugging Face (`dheerajnarne/textsummarizer`) for generating summaries.

---

## Key Concepts

### **Why T5?**
T5 stands out because of its text-to-text framework, making it ideal for tasks like summarization where both input and output are textual.

### **Dataset Details**
The chosen dataset (`antash420/text-summarization-alpaca-format`) has the following structure:
- **Input**: Text passages/articles.
- **Output**: Corresponding concise summaries.

This structure ensures compatibility with T5's input-output tokenization requirements.

### **Metrics**
The evaluation uses the **ROUGE** metric, which measures the overlap of n-grams between the generated and reference summaries. Key metrics include:
- **ROUGE-1**: Measures unigram overlap.
- **ROUGE-2**: Measures bigram overlap.
- **ROUGE-L**: Measures the longest matching sequence overlap.

---

## Results

The model demonstrates:
- Accurate extraction of key points from input texts.
- Coherent and concise summary generation.

Results can be further improved by experimenting with:
- Hyperparameter tuning.
- Larger model variants (e.g., T5-Base, T5-Large).
- Additional training data.

---

## How to Run

### **Notebook**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the dependencies (see Installation section).
3. Open the notebook in Jupyter and run the cells step-by-step.

### **Streamlit Application**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the folder containing the Streamlit app script (e.g., `app.py`).
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in your browser to interact with the application.

---

## Limitations

- The project uses a smaller dataset; larger and more diverse datasets can improve generalization.
- The T5-Small model is lightweight but may lack the capacity for more complex summarization tasks.

---

## Future Work

- Expand the dataset to include more diverse text types.
- Experiment with larger T5 models or alternative architectures like Pegasus for summarization.
- Integrate the model into an API or web interface for real-world usage.
- Enhance the Streamlit app with additional features, such as language support and advanced customization options.

---

## Hugging Face Repository

The fine-tuned model is available on Hugging Face. You can explore and use it [here](https://huggingface.co/dheerajnarne/textsummarizer).

---

## Acknowledgements

- **Hugging Face** for the Transformers library and datasets.
- **Google Research** for the development of the T5 model.
- **Contributors** of the `text-summarization-alpaca-format` dataset.
