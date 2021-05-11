from flask import Flask, render_template, request
import csv

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = Flask(__name__)



def recommend_movies(user_movie):
    movie = pd.read_csv("moviedata.csv")
    features = ['keywords', 'cast', 'genres', 'director', 'tagline']
    for feature in features:
        movie[feature] = movie[feature].fillna('')
    
    def combine_features(row):
        try:
            return row['keywords'] + " "+row['cast']+" "+row['genres']+" "+row['director']+" "+row['tagline']
        except:
            print("Error:", row)
    
    movie["combined_features"] = movie.apply(combine_features, axis=1)

    def title_from_index(index):
        return movie[movie.index == index]["title"].values[0]


    def index_from_title(title):
        title_list = movie['title'].tolist()
        common = difflib.get_close_matches(title, title_list, 1)
        titlesim = common[0]
        return movie[movie.title == titlesim]["index"].values[0]

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(movie["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)

    movie_index = index_from_title(user_movie)

    similar_movies = list(enumerate(cosine_sim[movie_index]))
    similar_movies_sorted = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)
    i = 0
    recommended_movies = []
    for rec_movie in similar_movies_sorted:
        if(i != 0):
            recommended_movies.append(title_from_index(rec_movie[0]))
        i = i+1
        if i > 10:
            break
    return recommended_movies


    



@app.route('/',  methods=['POST', 'GET'])
def index():
    if request.method == "GET":
        return render_template('index.html', movies=[])

    elif request.method == "POST":    
        user_movie = request.form.getlist('movie_title')[0]
        recommended_movies = recommend_movies(user_movie)        
        return render_template('index.html', movies=recommended_movies)


@app.route('/details/<movie_title>')
def details(movie_title):
    detail_list = {}
    movie = pd.read_csv("moviedata.csv")
    features = ['keywords', 'cast', 'genres', 'director', 'tagline']
    for feature in features:
        movie[feature] = movie[feature].fillna('')

    def index_from_title(title):
        title_list = movie['title'].tolist()
        common = difflib.get_close_matches(title, title_list, 1)
        titlesim = common[0]
        return movie[movie.title == titlesim]["index"].values[0]

    index = index_from_title(movie_title)
    detail_list['overview'] = movie.loc[index]['overview']
    detail_list['director'] = movie.loc[index]['director']
    detail_list['cast'] = movie.loc[index]['cast']
    detail_list['release_date'] = movie.loc[index]['release_date']
    detail_list['genres'] = movie.loc[index]['genres']
    detail_list['status'] = movie.loc[index]['status']
    recommended_movies = recommend_movies(movie_title)
    return render_template('detail.html', recommendations=recommended_movies, movie=movie_title, overview=detail_list)


if __name__ == '__main__':
    app.run(debug=True)
