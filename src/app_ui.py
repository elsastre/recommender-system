"""Aplicación Streamlit para sistema de recomendación basado en filtrado colaborativo neural.

Esta interfaz permite ingresar un ID de usuario y especificar la cantidad de recomendaciones
deseadas. Consume una API local para obtener recomendaciones personalizadas y las muestra
con sus respectivos puntajes de confianza.
"""

import streamlit as st
import requests

st.title("🎬 ML Recommender System")
st.markdown("Basado en **Neural Collaborative Filtering**")

user_id = st.number_input("Ingresa tu User ID:", min_value=1, value=1)
k = st.slider("¿Cuántas recomendaciones quieres?", 1, 20, 5)

if st.button("Obtener Recomendaciones"):
    # Llamamos a nuestra propia API (que debe estar corriendo en Docker o local)
    response = requests.get(f"http://localhost:8000/recommend/{user_id}?k={k}")

    if response.status_code == 200:
        data = response.json()
        for rec in data['recommendations']:
            st.write(f"**#{rec['rank']}** - {rec['title']} (Score: {rec['confidence_score']:.2f})")
    else:
        st.error("Usuario no encontrado.")