"""
Streamlit application for the NCF Recommender System.
Enterprise Edition: Clean UI, professional terminology, and seamless embedding discovery.
"""

import streamlit as st
import requests
import json
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NCF Recommender Engine", layout="wide", initial_sidebar_state="collapsed")

# ¡ESTA ES LA LÍNEA MÁGICA PARA DOCKER!
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Minimalist CSS for professional look
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .movie-title { font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; line-height: 1.2; }
        .sub-title { font-size: 0.75rem; color: #a1a1aa; }
        img {
            border-radius: 4px;
            transition: transform 0.2s ease;
        }
        img:hover { transform: scale(1.03); }
        .stButton>button {
            border: 1px solid #3f3f46;
            color: #d4d4d8;
            background-color: transparent;
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
        }
        .stButton>button:hover {
            border-color: #e4e4e7;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'recs_data' not in st.session_state:
    st.session_state.recs_data = None
    st.session_state.hist_data = None
    st.session_state.is_cold_start = False
if 'target_movie_id' not in st.session_state:
    st.session_state.target_movie_id = None
if 'target_movie_title' not in st.session_state:
    st.session_state.target_movie_title = None
if 'sim_cache' not in st.session_state:          # Caché para resultados de similitud
    st.session_state.sim_cache = {}

# --- HEADER SECTION ---
st.title("Neural Collaborative Filtering Engine")
st.markdown("Inference dashboard for personalized user recommendations and item-to-item similarity computation.")

col_input1, col_input2, col_btn = st.columns([2, 1, 1])
with col_input1:
    user_id = st.number_input("Target User ID", min_value=1, value=999999,
                              help="Use existing IDs (e.g., 1) or a non-existent ID (e.g., 999999) to test Cold Start policy.")
with col_input2:
    k = st.number_input("Top-K Candidates", min_value=1, max_value=10, value=5,
                        help="Number of items to retrieve from the model.")
with col_btn:
    st.write("<br>", unsafe_allow_html=True)
    if st.button("Run Inference", type="primary", use_container_width=True):
        with st.spinner('Computing predictions...'):
            try:
                # ACÁ REEMPLAZAMOS LOCALHOST POR LA VARIABLE API_URL
                res_hist = requests.get(f"{API_URL}/user/{user_id}/history?limit=6")
                res_recs = requests.get(f"{API_URL}/recommend/{user_id}?k={k}")
                
                if res_recs.status_code == 200 and res_hist.status_code == 200:
                    st.session_state.recs_data = res_recs.json()
                    st.session_state.hist_data = res_hist.json()
                    st.session_state.is_cold_start = st.session_state.recs_data.get("is_cold_start", False)
                    # Reset similarity view on new inference
                    st.session_state.target_movie_id = None
                else:
                    st.error("Inference service returned an error.")
            except requests.exceptions.ConnectionError:
                st.error("Connection Refused. Ensure the FastAPI backend is running.")

st.divider()

# --- MAIN DASHBOARD VIEWS ---
tab_inference, tab_metrics = st.tabs(["Inference Results", "System Metrics"])

if st.session_state.recs_data:
    
    # --- TAB 1: INFERENCE & SIMILARITY ---
    with tab_inference:
        # User Context (History)
        if st.session_state.is_cold_start:
            st.info(f"Cold Start Policy Activated: {st.session_state.recs_data.get('message')}")
        else:
            st.markdown("#### Historical Interactions")
            hist_list = st.session_state.hist_data.get('history', [])
            if hist_list:
                cols_hist = st.columns(max(len(hist_list), 6))
                for i, movie in enumerate(hist_list):
                    with cols_hist[i]:
                        st.image(movie['poster_url'], use_column_width=True)
                        st.markdown(f"<div class='sub-title'>Rated: {movie['rating']}</div>", unsafe_allow_html=True)
            else:
                st.caption("No historical interaction data available for this user.")
                
        st.write("<br>", unsafe_allow_html=True)
        
        # --- PRIMARY RECOMMENDATIONS (CON CHUNKING PARA EVITAR OVERLAP) ---
        st.markdown("#### Recommended Candidates" if not st.session_state.is_cold_start else "#### Global Baseline (Top Hits)")
        
        recs_list = st.session_state.recs_data.get('recommendations', [])
        
        # Dividimos las recomendaciones en grupos de 5 (filas)
        for i in range(0, len(recs_list), 5):
            chunk = recs_list[i:i+5]
            cols_recs = st.columns(5) # Forzamos siempre 5 columnas fijas para mantener el tamaño perfecto
            
            for j, rec in enumerate(chunk):
                with cols_recs[j]:
                    st.image(rec['poster_url'], use_column_width=True)
                    st.markdown(f"<div class='movie-title'>{rec['title']}</div>", unsafe_allow_html=True)
                    
                    if not st.session_state.is_cold_start:
                        st.progress(rec['confidence_score'])
                    
                    # Llave única anti-colisiones
                    if st.button("More like this", key=f"sim_main_{rec['movie_id']}_{i}_{j}", use_container_width=True):
                        st.session_state.target_movie_id = rec['movie_id']
                        st.session_state.target_movie_title = rec['title']
        else:
            if not recs_list:
                st.caption("No recommendations available.")

        # Embedding Similarity Results (usa caché para evitar llamadas repetidas)
        if st.session_state.target_movie_id:
            st.write("<br>", unsafe_allow_html=True)
            st.markdown(f"#### Similarity Vector Search: {st.session_state.target_movie_title}")
            
            # Verificar si ya tenemos en caché
            movie_id = st.session_state.target_movie_id
            if movie_id in st.session_state.sim_cache:
                sim_recs = st.session_state.sim_cache[movie_id]
            else:
                with st.spinner('Calculating cosine similarity...'):
                    try:
                        # ACÁ REEMPLAZAMOS LOCALHOST POR LA VARIABLE API_URL
                        res_sim = requests.get(f"{API_URL}/movie/{movie_id}/similar?k=6")
                        if res_sim.status_code == 200:
                            sim_data = res_sim.json()
                            sim_recs = sim_data.get('recommendations', [])
                            # Guardar en caché
                            st.session_state.sim_cache[movie_id] = sim_recs
                        else:
                            st.error("Similarity service error.")
                            sim_recs = []
                    except Exception as e:
                        st.error("Similarity computation failed.")
                        sim_recs = []
            
            # Mostrar resultados (desde caché o recién obtenidos)
            if sim_recs:
                cols_sim = st.columns(len(sim_recs))
                for j, sim_rec in enumerate(sim_recs):
                    with cols_sim[j]:
                        st.image(sim_rec['poster_url'], use_column_width=True)
                        st.markdown(f"<div class='sub-title'>{sim_rec['title']}</div>", unsafe_allow_html=True)
            else:
                st.caption("No similar movies found.")

    # --- TAB 2: SYSTEM METRICS ---
    with tab_metrics:
        st.markdown("#### Architecture & Telemetry")
        
        history_path = "models/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Inference Latency", f"{st.session_state.recs_data['inference_time_ms']} ms")
            col_kpi2.metric("Validation MAE", f"{history['val_mae'][-1]:.4f}")
            col_kpi3.metric("Training Epochs", f"{len(history['loss'])}")
            
            st.divider()
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Learning Curve (MSE)**")
                st.line_chart(pd.DataFrame({'Train Loss': history['loss'], 'Val Loss': history['val_loss']},
                                           index=range(1, len(history['loss']) + 1)))
                
            with col_chart2:
                st.markdown("**Global Rating Distribution**")
                data_path = 'data/ratings.dat'
                if os.path.exists(data_path):
                    df_ratings = pd.read_csv(data_path, sep='::', engine='python',
                                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                             encoding='latin-1')
                    st.bar_chart(df_ratings['rating'].value_counts().sort_index())
                else:
                    st.warning("Dataset not found at data/ratings.dat")
        else:
            st.warning("Telemetry data not found. Ensure the model has been trained.")