import numpy as np
import pandas as pd

class CollaborativeBasedRecommender():
    def _init_(self):
        self.train_data = None
        self.user_id = None
        self.song_id = None
        self.cooc_matrix = None
        self.artists = None
        self.song_artist_dict = None
      
    # retrive song preferences for a given user
    def get_user_songs(self, user):
        data = self.train_data[self.train_data[self.user_id] == user]
        songs = list(data[self.song_id].unique())
        return songs
    
    # retrieve users who like a given song
    def get_song_users(self, song):
        data = self.train_data[self.train_data[self.song_id] == song]
        users = set(data[self.user_id].unique())
        return users
    
    def create_cooc_matrix(self, user_songs, all_songs):
        # get users for each song from user's songs
        songs_users = []
        for i in range(0, len(user_songs)):
            songs_users.append(self.get_song_users(user_songs[i]))
        
        # initialize cooccurence matrix
        user_song_len = len(user_songs)
        all_song_len = len(all_songs)
        cooccurence_matrix = np.matrix(np.zeros(shape=(user_song_len, all_song_len)), float)
        
        for i in range(all_song_len):
            #find unique users of song i
            songs_data = self.train_data[self.train_data[self.song_id] == all_songs[i]]
            users = set(songs_data[self.user_id].unique())
            
            for j in range(user_song_len):       
                    
                #find unique users of song j
                users_2 = songs_users[j]
                    
                #find intersection of listeners of both songs
                users_inter = users.intersection(users_2)
                
                #compute cooccurence_matrix[i,j] if intersection exists
                if len(users_inter) != 0:
                    #compute ratio users who listen to both songs to those who listen to either
                    cooccurence_matrix[j,i] = float(len(users_inter))/float(len(users.union(users_2)))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        self.cooc_matrix = cooccurence_matrix
        return cooccurence_matrix
    
    def get_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non-zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        avgs = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_scores = np.array(avgs)[0].tolist()
        
        sort_index = sorted(((a,b) for b,a in enumerate(list(user_scores))), reverse=True)
        #HERE
        df = pd.DataFrame(columns=['user_id', 'song', 'artist', 'score'])
        
        # recommend top 10 songs
        rank = 1 
        for i in range(len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                #HERE
                df.loc[len(df)]=[user, all_songs[sort_index[i][1]], self.song_artist_dict[all_songs[sort_index[i][1]]], sort_index[i][0]]
                rank += 1
        
        #if no recommendations
        if df.shape[0] == 0:
            print("The user has no recommended songs.")
            return -1
        else:
            return df
        
    def create_model(self, user_id, song_id, train_data, artists):
        self.user_id = user_id
        self.song_id = song_id
        self.train_data = train_data
        self.artists = artists
    
    #print recommendation based on user preferences
    def make_recommendation(self, user):
        #retrieve songs of input user
        user_songs = self.get_user_songs(user)     
        print("Number of songs for user: %d" % len(user_songs))
        
        #retrieve unique songs from training data
        #all_songs = list(self.train_data[self.song_id].unique())
        all_songs = list(self.train_data[self.song_id])
        print("Number of songs in the training data: %d" % len(all_songs))
        artists_test = list(self.train_data[self.artists])
        print("Number of artists in training data: %d" % len(artists_test))
        
        #create song artist dictionary
        song_artist_dict = {}
        for i in range(len(all_songs)):
            song_artist_dict[all_songs[i]] = artists_test[i]
        
        self.song_artist_dict = song_artist_dict
        all_songs = list(song_artist_dict.keys())
        
        cooccurence_matrix = self.create_cooc_matrix(user_songs, all_songs)
        self.cooc_matrix = cooccurence_matrix
        
        return self.get_recommendations(user, cooccurence_matrix, all_songs, user_songs)
    
    def get_similar_songs(self, songs):
        all_songs = list(self.train_data[self.song].unique())
        print("Number of songs in the training data: %d" % len(all_songs))
        
        cooccurence_matrix = self.create_cooc_matrix(songs, all_songs)
        
        user = ""
        df = self.get_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df
        