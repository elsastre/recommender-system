"""Streamlit application for a recommender system based on Neural Collaborative Filtering.

This interface allows entering a user ID and specifying the number of desired
recommendations. It consumes a local API to fetch personalized recommendations
and displays them along with their confidence scores.
"""

import streamlit as st
import requests

st.title("🎬 ML Recommender System")
st.markdown("Based on **Neural Collaborative Filtering**")

user_id = st.number_input("Enter your User ID:", min_value=1, value=1)
k = st.slider("How many recommendations do you want?", 1, 20, 5)

if st.button("Get Recommendations"):
    # Call our own API (should be running locally or in Docker)
    response = requests.get(f"http://localhost:8000/recommend/{user_id}?k={k}")

    if response.status_code == 200:
        data = response.json()
        for rec in data['recommendations']:
            st.write(f"**#{rec['rank']}** - {rec['title']} (Score: {rec['confidence_score']:.2f})")
    else:
        st.error("User not found.")