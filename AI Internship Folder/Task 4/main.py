import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load MovieLens dataset
# Example data format: userId, movieId, rating
data = {
    'userId': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
    'movieId': [1, 2, 3, 1, 2, 4, 3, 4, 2, 3, 4],
    'rating': [4, 5, 2, 5, 3, 4, 2, 5, 4, 3, 4]
}
df = pd.DataFrame(data)

# Load data into Surprise library
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = train_test_split(dataset, test_size=0.25, random_state=42)

# Build and train the SVD model
model = SVD()
model.fit(trainset)

# Test the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Function to get movie recommendations
def get_recommendations(user_id, n_recommendations=5):
    # Get a list of all movie IDs
    all_movie_ids = df['movieId'].unique()
    rated_movie_ids = df[df['userId'] == user_id]['movieId']
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    # Predict ratings for unrated movies
    predicted_ratings = []
    for movie_id in unrated_movie_ids:
        predicted_rating = model.predict(user_id, movie_id).est
        predicted_ratings.append((movie_id, predicted_rating))

    # Sort predictions by rating
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Return top N recommendations
    recommended_movie_ids = [movie_id for movie_id, rating in predicted_ratings[:n_recommendations]]
    return recommended_movie_ids

# Example: Get recommendations for user 1
user_id = 1
recommendations = get_recommendations(user_id)
print(f"Top recommendations for user {user_id}: {recommendations}")