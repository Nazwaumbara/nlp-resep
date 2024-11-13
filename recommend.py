import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

vectorizer = joblib.load('C:/Users/ASUS/Downloads/NLP/tfidf_vectorizer_model.sav')
ingredient_vectors = joblib.load('C:/Users/ASUS/Downloads/NLP/ingredient_vectors.sav')
data = pd.read_csv('C:/Users/ASUS/Downloads/NLP/cleaned_dataset.csv')

def recommend_recipes(input_ingredients, top_n=5):
    input_vector = vectorizer.transform([input_ingredients])
    cosine_similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()
    related_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = data.iloc[related_indices]
    return recommendations[['Title', 'Ingredients']]

input_ingredients = "udang, bawang putih, garam, cabai, timun, ikan"
recommended_recipes = recommend_recipes(input_ingredients)
print(recommended_recipes)
