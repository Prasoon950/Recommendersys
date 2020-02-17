
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np
from flask import Flask, request, jsonify, render_template


data = pd.read_csv(r'C:\Users\DELL\Desktop\movie recommender\movies_metadata.csv')
data['title'] = data['title'].str.lower() 


data['tagline'] = data['tagline'].fillna('')
data['description'] = data['overview'] + data['tagline']
data['description'] = data['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]

datas = data
datas = datas.reset_index()
titles = datas['title']
indices = pd.Series(datas.index, index=datas['title'])


def get_recommendations(title):
    if title not in data['title'].unique():
        return('This movie is not in our database.')
    else:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]


get_recommendations('TOY STORY')



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('movie.html')
    
@app.route('/recommend')
def recommend():
    title = request.args.get('movie')
    r = get_recommendations(title)
    
    if type(r)==type('string'):
        return render_template('recommend.html',r=r,t='s')
    else:
        return render_template('recommend.html',r=r,t='l')



if __name__ == '__main__':
    app.run()