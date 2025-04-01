import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, diags
import json

class SimpleProfileRecommender:
    def __init__(self, data_path=None):
        self.scaler = MinMaxScaler()
        self.job_encoder = OneHotEncoder(handle_unknown='ignore')
        self.sex_encoder = OneHotEncoder(handle_unknown='ignore')
        self.language_encoder = OneHotEncoder(handle_unknown='ignore')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.user_profiles = None
        self.usernames = None
        
        self.weights = {
            'age': 1.0,
            'job': 0.5,
            'sex': 0.4,
            'language': 0.35,
            'interests': 0.9,
            'other_interests': 0.8
        }
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and preprocess the data from JSON"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        df = df[['username', 'age', 'job', 'sex', 'interests', 'other_interests', 'language']].copy()
        
        df['interests'] = df['interests'].fillna('').astype(str)
        df['other_interests'] = df['other_interests'].fillna('').astype(str)
        df['job'] = df['job'].fillna('unknown')
        df['sex'] = df['sex'].fillna('unknown')
        df['language'] = df['language'].fillna('unknown')
        
        self.usernames = df['username'].values
        
        df['age'] = self.scaler.fit_transform(df[['age']])
        
        job_encoded = self.job_encoder.fit_transform(df[['job']])
        sex_encoded = self.sex_encoder.fit_transform(df[['sex']])
        language_encoded = self.language_encoder.fit_transform(df[['language']])
        
        text_features = df['interests'] + ' ' + df['other_interests']
        tfidf_features = self.tfidf.fit_transform(text_features)
        
        self.user_profiles = hstack([
            df[['age']].values,
            job_encoded,
            sex_encoded,
            language_encoded,
            tfidf_features
        ])
    
    def get_recommendations(self, username, n_recommendations=5):
        """Get recommendations for a specific user"""
        if username not in self.usernames:
            return []
        
        user_idx = np.where(self.usernames == username)[0][0]
        
        weights = []
        weights.append(self.weights['age'])
        weights.extend([self.weights['job']] * len(self.job_encoder.categories_[0]))
        weights.extend([self.weights['sex']] * len(self.sex_encoder.categories_[0]))
        weights.extend([self.weights['language']] * len(self.language_encoder.categories_[0]))
        weights.extend([self.weights['interests']] * len(self.tfidf.get_feature_names_out()))
        
        weight_matrix = diags(weights)
        weighted_profiles = self.user_profiles.dot(weight_matrix)
        user_vector = weighted_profiles[user_idx]
        
        cosine_sim = cosine_similarity(user_vector, weighted_profiles).flatten()
        
        similar_users = cosine_sim.argsort()[-(n_recommendations+1):-1][::-1]
        
        recommendations = [self.usernames[idx] for idx in similar_users]
        return recommendations
