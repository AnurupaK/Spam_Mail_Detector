from flask import Flask,request,render_template
import pickle
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__,template_folder="templates",static_folder="static")

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('LogRModel.pkl','rb'))

@app.route('/')
def home():
    return render_template('email_layout.html')

@app.route('/spam_predict', methods=['POST'])
def predict():

    stem = PorterStemmer()

    def stemming(cleaned_text):
        stemmed_text = []
        for word in cleaned_text:
            if word not in stopwords.words('english'):
                stemmed_text.append(stem.stem(word))

        stemmed_text = ' '.join(stemmed_text)
        return stemmed_text

    def cleaning(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        stemmed_text = stemming(text)
        return stemmed_text

    content = request.form['mail']
    input_text = cleaning(content)


    X = vectorizer.transform([input_text])
    X_matrix = X.toarray()


    prediction = model.predict(X_matrix)
    print(prediction)

    if prediction[0]==1:
        result = "The mail is spam"
    else:
        result = "The mail is not spam"

    return render_template('email_layout.html', outcome=result,mail_content=content)


if __name__ == '__main__':
     app.run(debug=True)




