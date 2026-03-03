"""
TMDB API Client.

This module provides utility functions to interact with The Movie Database (TMDB) API,
specifically fetching movie posters based on movie titles.
"""

import os
import re
import logging
import requests
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("recommender-api.tmdb")

# Base URLs for the TMDB API and Image CDN
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


# ACÁ ESTÁ LA MAGIA: Guardamos en memoria RAM los últimos 5000 pósters
@lru_cache(maxsize=5000)
def get_movie_poster(title: str, api_key: str = None) -> str:
    """
    Fetch the poster URL for a given movie title from TMDB.

    Args:
        title (str): The movie title, potentially containing the release year 
                     (e.g., "Toy Story (1995)").
        api_key (str, optional): TMDB API Key. If not provided, it attempts to
                                 read it from the TMDB_API_KEY environment variable.

    Returns:
        str: The URL of the movie poster, or a fallback placeholder image 
             if not found or an error occurs.
    """
    # Fallback to environment variable if not passed explicitly
    key = api_key or os.getenv("TMDB_API_KEY")
    
    # Placeholder image in case of failure or missing API key
    placeholder_url = "https://via.placeholder.com/500x750?text=No+Poster"

    if not key:
        logger.warning("TMDB API Key is missing. Returning placeholder.")
        return placeholder_url

    # Clean the title: remove the year in parentheses " (1995)" to improve search hits
    clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()

    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": key,
        "query": clean_title,
        "language": "en-US",
        "page": 1
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("results") and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                
        # Return placeholder if the movie was queried but no poster exists
        return placeholder_url

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch poster for '{clean_title}': {e}")
        return placeholder_url
    
if __name__ == "__main__":
    print("Probando conexión con TMDB...")
    
    # Probamos con una película clásica del dataset de MovieLens
    test_title = "Toy Story (1995)" 
    poster_url = get_movie_poster(test_title)
    
    print(f"Película: {test_title}")
    print(f"Resultado URL: {poster_url}")