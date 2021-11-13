import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# data compiling
songs_num = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_19.csv", error_bad_lines=False)
songs_a = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_a.csv", error_bad_lines=False)
songs_b = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_b.csv", error_bad_lines=False)
songs_c = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_c.csv", error_bad_lines=False)
songs_d = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_d.csv", error_bad_lines=False)
songs_e = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_e.csv", error_bad_lines=False)
songs_f = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_f.csv", error_bad_lines=False)
songs_g = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_g.csv", error_bad_lines=False)
songs_h = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_h.csv", error_bad_lines=False)
songs_i = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_i.csv", error_bad_lines=False)
songs_j = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_j.csv", error_bad_lines=False)
songs_k = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_k.csv", error_bad_lines=False)
songs_l = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_l.csv", error_bad_lines=False)
songs_m = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_m.csv", error_bad_lines=False)
songs_n = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_n.csv", error_bad_lines=False)
songs_o = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_o.csv", error_bad_lines=False)
songs_p = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_p.csv", error_bad_lines=False)
songs_q = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_q.csv", error_bad_lines=False)
songs_r = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_r.csv", error_bad_lines=False)
songs_s = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_s.csv", error_bad_lines=False)
songs_t = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_t.csv", error_bad_lines=False)
songs_u = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_u.csv", error_bad_lines=False)
songs_v = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_v.csv", error_bad_lines=False)
songs_w = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_w.csv", error_bad_lines=False)
songs_x = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_x.csv", error_bad_lines=False)
songs_y = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_y.csv", error_bad_lines=False)
songs_z = pd.read_csv("azlyrics-scraper/azlyrics_lyrics_z.csv", error_bad_lines=False)

# compile data, clean columns
songs_list = [songs_num, songs_a, songs_b, songs_c, songs_d, songs_e, songs_f, songs_g, songs_h, songs_i, songs_j, songs_k, songs_l,
         songs_m, songs_n, songs_o, songs_p, songs_q, songs_r, songs_s, songs_t, songs_u, songs_v, songs_w, songs_x, songs_y, songs_z]

songs = pd.concat(songs_list, axis=0, ignore_index=True)
songs = songs.drop(["ARTIST_URL", "SONG_URL"], axis=1)
#songs.head()

#Sample 5000 songs
song_sample = songs.sample(n=5000).reset_index(drop=True)

# Initialize tfidf vectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')

# Fit and transform
lyrics_matrix = tfidf.fit_transform(song_sample["LYRICS"].values.astype('U'))

cosine_similarities = cosine_similarity(lyrics_matrix)

# find 50 most similar songs for each song in our dataset
similarities = {}
for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    similarities[song_sample['SONG_NAME'].iloc[i]] = [(cosine_similarities[i][x], song_sample['SONG_NAME'][x],
                                           song_sample['ARTIST_NAME'][x]) for x in similar_indices][1:]

# make rec
recommendations = ContentBasedRecommender(similarities)

recommendation = {
    "song": song_sample['SONG_NAME'].iloc[40],
    "number_songs": 4
}
recommendations.recommend(recommendation)