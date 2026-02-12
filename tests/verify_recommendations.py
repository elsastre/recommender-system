"""Validación de recomendaciones: historial del usuario y títulos generados.

Carga los títulos de películas desde u.item y las calificaciones desde u.data.
Proporciona una función para mostrar las películas mejor puntuadas por un usuario
junto con las recomendaciones producidas por el sistema, permitiendo verificar
si las sugerencias son relevantes.
"""

import pandas as pd

# Cargar títulos de películas
# El archivo u.item usa latin-1 y está separado por '|'
cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + ['genre' + str(i) for i in range(19)]
items = pd.read_csv('data/u.item', sep='|', names=cols, encoding='latin-1')
movie_titles = dict(zip(items['movie_id'], items['title']))

# Cargar ratings para ver el historial del usuario
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=ratings_cols)

def check_logic(user_id, recommended_ids):
    """Muestra el historial de un usuario y las recomendaciones generadas.

    Imprime en consola las cinco películas con mayor calificación del usuario
    seguido de las películas recomendadas, empleando los títulos reales.

    Args:
        user_id (int): Identificador del usuario a evaluar.
        recommended_ids (list): Lista de IDs de películas recomendadas.
    """
    print(f"--- HISTORIAL DEL USUARIO {user_id} (Top 5 favoritas) ---")
    user_history = ratings[ratings['user_id'] == user_id].sort_values(by='rating', ascending=False).head(5)
    for _, row in user_history.iterrows():
        print(f"- {movie_titles.get(row['item_id'])} (Rating: {row['rating']})")

    print(f"\n--- RECOMENDACIONES GENERADAS ---")
    for mid in recommended_ids:
        print(f"-> {movie_titles.get(mid)}")

# Tus resultados actuales
my_recs = [1449, 850, 1467, 483, 1306]
check_logic(1, my_recs)