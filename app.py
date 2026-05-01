# Generated from: app.ipynb
# Converted at: 2026-04-30T05:51:06.548Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import ollama
import plotly.express as px

st.set_page_config(page_title="Amazon Insight Engine", layout="wide")

@st.cache_resource
def load_assets():
    db = lancedb.connect("./amazon_reviews_db_PERMANENT")
    table = db.open_table("reviews_topic_updated")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return table, embed_model


table, embed_model = load_assets()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Universe")
selected_cat = st.sidebar.multiselect("Category", ["Books", "Music"], default=["Books", "Music"])
min_rating = st.sidebar.slider("Minimum Star Rating", 1, 5, 1)

st.title("🚀 Semantic Discovery & RAG Intelligence")

tab1, tab2 = st.tabs(["💬 AI Assistant (RAG)", "📊 Market Insights (Topics)"])

with tab1:
    st.subheader("Ask your Data")
    user_query = st.chat_input("e.g., What are the common complaints about recent albums?")

    if user_query:
        # 1. Retrieval
        query_vec = embed_model.encode(user_query).tolist()

        search_results = table.search(query_vec).where(f"starRating >= {min_rating} AND productCategory = {selected_cat} AND sentimentCategory = 'NEGATIVE'").limit(50).to_polars()

        context = "\n".join([f"- {r['reviewText']}" for r in search_results.to_dicts()])

        with st.chat_message("assistant"):
            # Simple RAG Prompt
            prompt = f"Answer based on these reviews:\n{context}\n\nQuestion: {user_query}"
            response = ollama.generate(model='llama3', prompt=prompt)
            st.write(response['response'])

        with st.expander("View Source Reviews"):
            st.dataframe(search_results.select(["productTitle", "starRating", "topic_label", "reviewText"]))

with tab2:
    st.subheader("Topic Distribution & Sentiment")
    
    # Load data for viz
    full_df = table.to_polars().collect()
    
    # Chart 1: Top Topics (The complaint_count logic)
    topic_counts = (
        full_df.group_by("topic_label")
        .count()
        .sort("count", descending=True)
        .head(10)
    )
    
    fig_topics = px.bar(
        topic_counts, 
        x="count", 
        y="topic_label", 
        orientation='h',
        title="Top 10 Discussion Themes",
        color_discrete_sequence=['#00CC96']
    )
    st.plotly_chart(fig_topics, use_container_width=True)

    # Chart 2: Sentiment vs Topic Heatmap
    sentiment_map = (
        full_df.group_by(["topic_label", "sentimentCategory"])
        .count()
        .pivot(index="topic_label", on="sentimentCategory", values="count")
        .fill_null(0)
    )
    st.write("Sentiment Breakdown by Topic")
    st.dataframe(sentiment_map)