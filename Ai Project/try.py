import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn import preprocessing 
# Load the data
data = pd.read_csv('ipl_data.csv')
label_encoder = preprocessing.LabelEncoder()


data["venue"]= label_encoder.fit_transform(data["venue"])
data["team1"]= label_encoder.fit_transform(data["team1"])
data["team2"]= label_encoder.fit_transform(data["team2"])
data["toss_winner"]= label_encoder.fit_transform(data["toss_winner"])
data["toss_decision"]= label_encoder.fit_transform(data["toss_decision"])
data["winner"]= label_encoder.fit_transform(data["winner"])
# Preprocess the data
X = data[['Year', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', '1st innings  win ratio', '2 nd innings win ratio', 'head to head ratio team1', 'head to head ratio team2']]
y = data['winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')