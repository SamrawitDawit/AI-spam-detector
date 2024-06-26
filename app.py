import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

df = pd.read_csv('./spam.csv', encoding='latin-1')
df['spam'] = df.v1.map({'ham':0, 'spam':1})
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'v1'], axis=1)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df.v2, df.spam, test_size=0.2)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_test_count = v.transform(X_test)
model = MultinomialNB()
model.fit(X_train_count, y_train)
ypred = model.predict(X_test_count)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    message = data['message']
    message_transformed = v.transform([message])
    prediction = model.predict(message_transformed)
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({'result': result})
if __name__ == '__main__':
    app.run(debug = True)

    


