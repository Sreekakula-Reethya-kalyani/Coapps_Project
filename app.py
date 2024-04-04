import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Function to generate synthetic rumors
def generate_rumors(num_samples):
    rumors = [
        "Aliens spotted in New York City!",
        "Eating carrots cures cancer, scientists say.",
        # Add more rumors as needed
    ]
    return np.random.choice(rumors, num_samples)

# Function to generate synthetic facts
def generate_facts(num_samples):
    facts = [
        "The sky is blue.",
        "Water boils at 100 degrees Celsius.",
        # Add more facts as needed
    ]
    return np.random.choice(facts, num_samples)

# Generate synthetic data
num_samples = 1000
rumors = generate_rumors(num_samples)
facts = generate_facts(num_samples)
data = np.concatenate([rumors, facts])
labels = np.array(['Rumor'] * num_samples + ['Fact'] * num_samples)

# Create DataFrame
df = pd.DataFrame({'Text': data, 'Label': labels})

# Display data
st.write("## Synthetic Rumors Dataset")
st.write(df)

# Preprocess the text data and vectorize it using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Text'])
y = df['Label']

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Function to predict using the classifier
def predict_rumor_or_fact(sentence):
    sentence_vectorized = vectorizer.transform([sentence])
    prediction = clf.predict(sentence_vectorized)
    return prediction[0]

# Streamlit UI
st.title("Rumor Mill: Tracking Viral Rumors through Textual Analysis")
user_input = st.text_input("Enter a sentence to classify as rumor or fact:")
if user_input:
    prediction = predict_rumor_or_fact(user_input)
    st.write(f"Prediction: {prediction}")

# Additional functionalities (Logistic Regression, K-Means Clustering) can be added similarly.

# Optionally, you can save the trained model if needed
# joblib.dump(clf, 'rumor_mill_classifier.pkl')
