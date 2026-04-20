import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self):
        # __init__ se ejecuta cuando creás la clase
        # Acá guardamos los datos en memoria
        self.movies = None          # DataFrame de películas
        self.ratings = None         # DataFrame de ratings
        self.cosine_sim = None      # Matriz de similitud (content-based)
        self.indices = None         # Índice título → posición
        self.user_movie_matrix = None  # Matriz usuario-película
        self.user_similarity = None    # Similitud entre usuarios
        self.svd = None             # Modelo SVD
    
    def train(self):
        # Carga datos y entrena el modelo
        self.movies = pd.read_csv('../data/movies_clean.csv')
        self.ratings = pd.read_csv('../data/ratings_clean.csv')

        # Paso 2 - Content-based: preparar géneros
        self.movies['genres_clean'] = self.movies['genres'].str.replace('|', ' ', regex=False)
        self.movies['genres_clean'] = self.movies['genres_clean'].str.replace('Sci-Fi', 'SciFi', regex=False)
        self.movies['genres_clean'] = self.movies['genres_clean'].str.replace('Film-Noir', 'FilmNoir', regex=False)

        # Paso 3 - Content-based: TF-IDF y similitud
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.movies['genres_clean'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Paso 4 - Índice de títulos
        self.indices = pd.Series(self.movies.index, index=self.movies['title'])

        # Paso 5 - Collaborative: matriz usuario-película
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values= 'rating'
        ).fillna(0)

        # Paso 6 - Sparse matrix + SVD
        matrix_sparse = csr_matrix(self.user_movie_matrix.values)
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        matrix_svd = self.svd.fit_transform(matrix_sparse)

        # Paso 7 - Similitud entre usuarios
        self.user_similarity = cosine_similarity(matrix_svd)
    
    def get_content_recommendations(self, title, n=10):
        # Paso 1 - Búsqueda parcial e insensible a mayúsculas
        matches = self.indices[self.indices.index.str.lower().str.contains(title.lower())]
        
        if len(matches) == 0:
            return "No se encontró ninguna película con ese título"
        
        # Paso 2 - Tomamos el primer resultado
        """
        matches es una Serie de pandas donde:
        El índice son los títulos → matches.index[0]
        Los valores son los índices numéricos → matches.iloc[0]
        """
        idx = matches.iloc[0]  # Índice numérico de la película encontrada
        # Obtenemos los scores de similitud con todas las demás
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Ordenamos por similitud descendente
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Tomamos las n más similares (excluimos la primera que es ella misma)
        sim_scores = sim_scores[1:n+1]

        # Obtenemos los índices de las películas
        movie_indices = [i[0] for i in sim_scores]
        return self.movies[['title', 'genres']].iloc[movie_indices]
    

    def get_collaborative_recommendations(self, user_id, n=10):
        # Paso 1 - Índice del usuario en la matriz
        user_idx = list(self.user_movie_matrix.index).index(user_id)
        
        # Paso 2 - Usuarios más similares
        sim_scores = list(enumerate(self.user_similarity[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Top 5 usuarios similares
        
        # Paso 3 - Películas que vieron esos usuarios
        similar_users_idx = [i[0] for i in sim_scores]
        similar_users_ratings = self.user_movie_matrix.iloc[similar_users_idx]
        
        # Paso 4 - Películas que el usuario actual NO vio
        user_ratings = self.user_movie_matrix.iloc[user_idx]
        unseen_movies = user_ratings[user_ratings == 0].index
        
        # Paso 5 - Promedio de ratings de usuarios similares para películas no vistas
        recommendations = similar_users_ratings[unseen_movies].mean(axis=0)
        recommendations = recommendations[recommendations > 0]
        recommendations = recommendations.sort_values(ascending=False).head(n)
        
        # Paso 6 - Unimos con info de películas
        rec_movies = self.movies[self.movies['movieId'].isin(recommendations.index)]
        return rec_movies[['title', 'genres']]
