# Amazon Semantic Discovery & Sentiment Intelligence Platform

A production-grade MLE project that transforms 10,000+ unstructured Amazon reviews (Books & Music) into actionable business intelligence using Vector Search, Topic Modeling, and RAG.

## 🚀 The Mission
The goal of this project is to move beyond simple sentiment analysis. It provides a "Semantic Discovery" portal where users can search for product insights by meaning rather than keywords, identify the top 5 negative pain points automatically, and chat with an LLM grounded in verified review data.

---

## 🏗️ System Architecture

1.  **Data Engineering**: Ingested and preprocessed 10,000+ Amazon reviews using `Polars` for high-performance data manipulation.
2.  **Vector Infrastructure**: Implemented **LanceDB** for serverless, on-disk vector storage, enabling lightning-fast similarity search.
3.  **Topic Modeling**: Leveraged **BERTopic** (with Guided Seed Topics) to categorize reviews into business-relevant themes like "Shipping Quality," "Physical Condition," and "Product Content."
4.  **Intelligence Layer**: Integrated **Hugging Face Transformers** for sentiment scoring and **Groq (Llama 3.1)** for Retrieval-Augmented Generation (RAG).
5.  **UI/UX**: Developed a multi-tab dashboard in **Streamlit** to visualize topic distributions and serve the AI Assistant.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Database**: LanceDB (Vector DB)
*   **Analysis**: Polars, BERTopic, Plotly
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **LLM Engine**: Groq Cloud API (Llama 3.1) / Ollama
*   **Frontend**: Streamlit

---

## 🌟 Key Features

### 🔍 Semantic & Metadata Search
Users can query the dataset using natural language (e.g., *"Why are people upset about the shipping packaging?"*). The system uses vector embeddings to find relevant reviews even if specific keywords aren't present, combined with metadata filtering for star ratings.

### 📊 Automated Pain-Point Discovery
The dashboard identifies the "Top 5 Negative Topics" by filtering for dissatisfied users and aggregating topic frequency. This provides an immediate view of where a product or service is failing.

### 💬 Grounded RAG Chatbot
An AI assistant that answers questions based **ONLY** on the retrieved reviews. This prevents LLM hallucinations and ensures all answers are traceable to actual customer feedback.

---

## 🏃 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/amazon-sentiment-intelligence.git](https://github.com/yourusername/amazon-sentiment-intelligence.git)
cd amazon-sentiment-intelligence