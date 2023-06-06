from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the data from a CSV file
data = pd.read_csv('filtered_data.csv')

# Convert the text data into numerical features
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(data['book_name'])

# Train a logistic regression model on the full dataset
model = LogisticRegression()
model.fit(X_vect, data['is_ebook'])

# Initialize a Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['book_name']
    X_input = vectorizer.transform([title])
    y_pred = model.predict(X_input)[0]
    if y_pred == 1:
        prediction = 'an e-book'
    else:
        prediction = 'not an e-book'
    return f'The book "{title}" is {prediction}.'

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
