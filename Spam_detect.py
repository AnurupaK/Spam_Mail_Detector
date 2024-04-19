import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

mail = pd.read_csv('mail_data.csv')
df = mail.copy()

stem = PorterStemmer()
def stemming(cleaned_text):
    stemmed_text = []
    for word in cleaned_text:
        if word not in stopwords.words('english'):
            stemmed_text.append(stem.stem(word))

    stemmed_text = ' '.join(stemmed_text)
    return stemmed_text


def cleaning(text):
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    stemmed_text = stemming(text)
    return stemmed_text

df["Message"] = df["Message"].apply(cleaning)



encode = LabelEncoder()
Y_data = encode.fit_transform(df["Category"])
Y_data = pd.DataFrame(Y_data,columns=['Target'])

for i,j in zip(mail["Category"],Y_data["Target"]):
    print(i,j)

exit()


vectorizer = TfidfVectorizer()


X_data = df["Message"]

X_matrix = vectorizer.fit_transform(X_data)
X_matrix = X_matrix.toarray()
features = vectorizer.get_feature_names_out()

X_df = pd.DataFrame(X_matrix,columns=features)
df_main = pd.concat([X_df,Y_data],axis=1)


X = df_main.drop(columns='Target',axis=1)
Y = df_main['Target']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

LogRModel = LogisticRegression()
LogRModel.fit(Xtrain,Ytrain)

##Accuracy on Train
Ytrain_predict = LogRModel.predict(Xtrain)
accuracy_train = accuracy_score(Ytrain_predict,Ytrain)

##Accuracy on Train
Ytest_predict = LogRModel.predict(Xtest)
accuracy_test = accuracy_score(Ytest_predict,Ytest)

print(f"Accuracy on train data is:{accuracy_train} | Accuracy on test data is:{accuracy_test}")


for i,j in zip(Ytrain_predict,Ytrain):
   print(f'Predict:{i} | Actual:{j}')



# pickle.dump(LogRModel,open('LogRModel.pkl','wb'))
# LogRModel = pickle.load(open('LogRModel.pkl','rb'))
#
# pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
# vectorizer = pickle.load(open('vectorizer.pkl','rb'))