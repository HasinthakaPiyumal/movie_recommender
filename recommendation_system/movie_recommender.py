import pandas as pd 
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movie_data_path = 'movies.csv'
# movie_data_path = 'TMDB_movie_dataset_v11.csv'


class MovieRecommender:
    def __init__(self):
        self.movie_data = pd.read_csv(movie_data_path)
        self.selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        # self.selected_features = ['genres', 'keywords', 'tagline',  'overview', 'spoken_languages']
        for feature in self.selected_features:
            self.movie_data[feature] = self.movie_data[feature].fillna('')
        
        combined_features = self.movie_data.genres + self.movie_data.keywords + self.movie_data.tagline + self.movie_data.cast + self.movie_data.director
        # combined_features = self.movie_data.genres + self.movie_data.keywords + self.movie_data.tagline + self.movie_data.overview + self.movie_data.spoken_languages
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        self.similarity = cosine_similarity(feature_vectors)

    def find_movie_index(self, movie_name):
        list_of_all_titles = self.movie_data.title.tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        if not find_close_match:
            return False
        
        close_match = find_close_match[0]
        index_of_movie = self.movie_data[self.movie_data.title == close_match].index.values[0]
        return index_of_movie

    def get_recommended_movies(self, movie_name, n_recommendations=10):
        if not movie_name:
            return "Please provide a movie name."
        index_of_movie = self.find_movie_index(movie_name)
        if index_of_movie is False:
            return "Movie not found in the database."
        
        similarity_score = list(enumerate(self.similarity[index_of_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        top_similar_movies_index = sorted_similar_movies[:n_recommendations]
        top_similar_movies = []
        
        for movie_index in top_similar_movies_index:
            movie = self.movie_data[self.movie_data.index == movie_index[0]]['title'].values[0]
            top_similar_movies.append(movie)
        
        return top_similar_movies

if __name__ == "__main__":
    movie_name = input('Your favourite movie: ')
    mr = MovieRecommender()
    print(mr.get_recommended_movies(movie_name))