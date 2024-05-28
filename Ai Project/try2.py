import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
df=pd.read_csv("ipl_data.csv")

# Ignores warnings 
import warnings
warnings.filterwarnings("ignore")

"""
df=df.dropna() 
df = df[(np.abs(stats.zscore(df.select_dtypes(exclude='object'))) < 3).all(axis=1)]

st_x=StandardScaler()
X=st_x.fit_transform(X)

"""
label_encoder = preprocessing.LabelEncoder()


df["venue"]= label_encoder.fit_transform(df["venue"])
df["team1"]= label_encoder.fit_transform(df["team1"])
df["team2"]= label_encoder.fit_transform(df["team2"])
df["toss_winner"]= label_encoder.fit_transform(df["toss_winner"])
df["toss_decision"]= label_encoder.fit_transform(df["toss_decision"])
df["winner"]= label_encoder.fit_transform(df["winner"])



X=df.iloc[:,0:10]
y=df.iloc[:,10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


from sklearn.neighbors import KNeighborsClassifier
print("KNN")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


from sklearn.naive_bayes import MultinomialNB
print("Naive Bayes")
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

from sklearn.svm import SVC
print("SVM")
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


from sklearn.linear_model import LogisticRegression
print("LogisticRegression")
model = LogisticRegression(max_iter=50)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

from sklearn.ensemble import RandomForestClassifier
print("Random Forest")
model = RandomForestClassifier(max_features=8,max_depth=3,criterion="entropy",random_state=137)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
