import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

data = pd.read_csv('C:/Users/ASUS/Downloads/NLP/gabungan_dataset.csv')

data = data.dropna(subset=['Ingredients'])

def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower().strip()
    return text

data['Ingredients'] = data['Ingredients'].apply(clean_text)

vectorizer = TfidfVectorizer()
ingredient_vectors = vectorizer.fit_transform(data['Ingredients'])

joblib.dump(vectorizer, 'tfidf_vectorizer_model.sav')
joblib.dump(ingredient_vectors, 'ingredient_vectors.sav')

data.to_csv('cleaned_dataset.csv', index=False)

print("Model dan dataset telah diekspor.")
