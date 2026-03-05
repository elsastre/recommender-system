import os
import urllib.request
import zipfile

# URLs oficiales
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# TODO: Cuando crees el Release en GitHub con tu modelo, pegá el link acá
MODEL_URL = "https://github.com/elsastre/recommender-system/releases/download/v1.0.0/recommender_v1.keras"

def download_file(url, dest_path):
    print(f"Descargando: {url}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Guardado en: {dest_path}")

def main():
    print("Iniciando descarga de artefactos...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Bajar y extraer el dataset oficial
    zip_path = "data/ml-1m.zip"
    if not os.path.exists("data/ratings.dat"):
        download_file(MOVIELENS_URL, zip_path)
        print("Extrayendo dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraemos los dat y los movemos a la raíz de data/
            zip_ref.extract("ml-1m/movies.dat", "data/")
            zip_ref.extract("ml-1m/ratings.dat", "data/")
        
        os.rename("data/ml-1m/movies.dat", "data/movies.dat")
        os.rename("data/ml-1m/ratings.dat", "data/ratings.dat")
        os.rmdir("data/ml-1m")
        os.remove(zip_path)
        print("✅ Dataset MovieLens 1M listo.")
    else:
        print("✅ Dataset ya existe. Saltando...")

    # 2. Bajar tu modelo entrenado
    model_path = "models/recommender_v1.keras"
    if not os.path.exists(model_path):
        try:
            download_file(MODEL_URL, model_path)
            print("✅ Modelo pre-entrenado listo.")
        except Exception as e:
            print(f"⚠️ Error al bajar el modelo (¿Actualizaste la URL?): {e}")
    else:
        print("✅ Modelo ya existe. Saltando...")

    print("\n🎉 ¡Entorno listo! Ya podés levantar Docker.")

if __name__ == "__main__":
    main()