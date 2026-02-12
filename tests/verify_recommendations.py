"""Recommendation validation: user history and generated titles.

Loads movie titles from u.item and ratings from u.data. Provides a helper to
display a user's highest-rated movies alongside the recommendations produced
by the system, allowing verification of suggestion relevance.
"""

import pandas as pd

# Load movie titles
# The u.item file uses latin-1 encoding and is pipe-separated ('|')
cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + ['genre' + str(i) for i in range(19)]
items = pd.read_csv('data/u.item', sep='|', names=cols, encoding='latin-1')
movie_titles = dict(zip(items['movie_id'], items['title']))

# Load ratings to inspect the user's history
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=ratings_cols)

def check_logic(user_id, recommended_ids):
    """Show a user's history and the generated recommendations.

    Prints the user's top five rated movies followed by the recommended movies,
    using real titles.

    Args:
        user_id (int): Identifier of the user to evaluate.
        recommended_ids (list): List of recommended movie IDs.
    """
    print(f"--- USER HISTORY {user_id} (Top 5 favorites) ---")
    user_history = ratings[ratings['user_id'] == user_id].sort_values(by='rating', ascending=False).head(5)
    for _, row in user_history.iterrows():
        print(f"- {movie_titles.get(row['item_id'])} (Rating: {row['rating']})")

    print(f"\n--- GENERATED RECOMMENDATIONS ---")
    for mid in recommended_ids:
        print(f"-> {movie_titles.get(mid)}")

# Your current results
my_recs = [1449, 850, 1467, 483, 1306]
check_logic(1, my_recs)