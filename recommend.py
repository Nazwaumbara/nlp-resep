import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

vectorizer = joblib.load('D:/NLP/tfidf_vectorizer_model.sav')
ingredient_vectors = joblib.load('D:/NLP/ingredient_vectors.sav')
data = pd.read_csv('D:/NLP/cleaned_dataset.csv')

def recommend_recipes(input_ingredients, top_n=8):
    input_vector = vectorizer.transform([input_ingredients])
    cosine_similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()
    related_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = data.iloc[related_indices]
    return recommendations[['Title', 'Ingredients', 'Steps']]

input_ingredients = "udang, bawang putih, garam, cabai, timun, ikan"
recommended_recipes = recommend_recipes(input_ingredients)
print(recommended_recipes)
