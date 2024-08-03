import streamlit as st
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\\Users\\Saravanan\\OneDrive\\Desktop\\ipl\\flipkart\\flipkart_data.csv")

# Preprocess dataset
df['label'] = df['rating'].apply(lambda x: 1 if x >= 5 else 0)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                          for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in stop_words))
    return preprocessed_text

df['review'] = preprocess_text(df['review'].values)

# Create the word cloud for positive reviews
consolidated = ' '.join(word for word in df['review'][df['label'] == 1].astype(str))
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
wordCloud.generate(consolidated)

# Vectorize reviews
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(df['review']).toarray()
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Evaluate model on test set
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

# Define Streamlit app
st.title("Flipkart Review Sentiment Analysis")

# User input
review_input = st.text_area("Enter the review:")
rating_input = st.number_input("Enter the rating:", min_value=1, max_value=10, value=5)

# Preprocess user input
review_input_processed = ' '.join(token.lower() for token in nltk.word_tokenize(re.sub(r'[^\w\s]', '', review_input)) if token.lower() not in stop_words)
review_vectorized = cv.transform([review_input_processed]).toarray()

# Predict sentiment
if st.button("Predict Sentiment"):
    prediction = model.predict(review_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"The review sentiment is: {sentiment}")

# Show model performance on test set
st.subheader("Model Performance on Test Set")
st.write(f"Accuracy: {test_accuracy:.2f}")

# Show the confusion matrix for the test data
st.subheader("Confusion Matrix on Test Data")
cm = confusion_matrix(y_test, test_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
fig, ax = plt.subplots()
cm_display.plot(ax=ax)
st.pyplot(fig)

# Show the word cloud for positive reviews
st.subheader("Word Cloud for Positive Reviews")
fig_wordcloud, ax_wordcloud = plt.subplots()
ax_wordcloud.imshow(wordCloud, interpolation='bilinear')
ax_wordcloud.axis('off')
st.pyplot(fig_wordcloud)
