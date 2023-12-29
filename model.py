import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sc


## prevent scientific numbers
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:16.3f}'.format}, linewidth=130)


df1 = pd.read_csv(r"./data_train_hw4_problem1.csv" ,encoding='latin-1')
df1.info()
df1.head(10)
df1.isna().sum().sum()
df1['spam'].value_counts()
## if spam is true, then spam is converted to 1
df1 = df1.replace({True:1, False:0})
df1.head(10)
from sklearn.feature_extraction.text import CountVectorizer
df1_x = df1["text"]
df1_y = df1["spam"]

print(df1_x)
cv = CountVectorizer()
X = cv.fit_transform(df1_x)

X_train = X
y_train = df1_y
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train,y_train)
df2 = pd.read_csv(r"./data_test_hw4_problem1.csv" ,encoding='latin-1')
print(df2)

str(df2)
X_test = cv.transform(df2["text"])

nb_pred = nb_model.predict(X_test)

print(nb_pred)
predictions = np.array(nb_pred, dtype=bool)

print(predictions)
pred = pd.DataFrame(data = predictions, columns = ["spam"])

pred.to_csv("predictions.csv")