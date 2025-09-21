import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv('medicines.csv')

# Combine all symptom-disease mappings
df['features'] = df['symptom'] + ' ' + df['disease']
X = df['features']
y = df['medicine']

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

# Predict from user input
def recommend_medicine(user_input):
    with open("model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    input_vec = vectorizer.transform([user_input.lower()])
    prediction = model.predict(input_vec)
    return prediction[0]

# User input
user_input = input("Enter your symptoms (e.g. 'fever cough'):\n")
print("Recommended Medicine:", recommend_medicine(user_input))
