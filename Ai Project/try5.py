import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree 
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("ipl_data.csv")

df.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Delhi Capitals',
                 'Chennai Super Kings','Punjab Kings','Rajasthan Royals', 
                  'Gujarat Titans', 'Sunrisers Hyderabad',
                 'Lucknow Super Giants'],
                 ['MI', 'KKR', 'RCB', 'DC', 'CSK','PBKS', 'RR', 'GT','SRH','LSG'], inplace=True)

encode = {'team1': {'MI':0, 'KKR':1, 'RCB':2, 'DC':3, 'CSK':4,'PBKS':5, 'RR':6, 'GT':7,'SRH':8,'LSG':9}, 
          'team2': {'MI':0, 'KKR':1, 'RCB':2, 'DC':3, 'CSK':4,'PBKS':5, 'RR':6, 'GT':7,'SRH':8,'LSG':9},
          'toss_winner': {'MI':0, 'KKR':1, 'RCB':2, 'DC':3, 'CSK':4,'PBKS':5, 'RR':6, 'GT':7,'SRH':8,'LSG':9},
          'winner': {'MI':0, 'KKR':1, 'RCB':2, 'DC':3, 'CSK':4,'PBKS':5, 'RR':6, 'GT':7,'SRH':8,'LSG':9},
          'venue': {'Rajiv Gandhi International Stadium, Hyderabad':0,
                    'M Chinnaswamy Stadium, Bengaluru':1, 'Wankhede Stadium, Mumbai':2,
                     'Holkar Cricket Stadium, Indore':3, 'Eden Gardens, Kolkata ':4,
                     'Arun Jaitley Stadium, Delhi':5,
                     'Punjab Cricket Association Stadium, Mohali':6,
                     'MA Chidambaram Stadium, Chepauk, Chennai':7,
                     'Sawai Mansingh Stadium, Rajasthan':8,
                     'Maharashtra Cricket Association Stadium, Pune':9,
                     'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':10,
                     'Brabourne Stadium, Mumbai':11, 'Dr DY Patil Sports Academy, Mumbai':12,
                     'Narendra Modi Stadium, Ahmedabad':13,
                     'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow':14,
                     'Barsapara Cricket Stadium, Guwahati':15,
                     'Himachal Pradesh Cricket Association Stadium, Dharamshala':16},
            'toss_decision':{'Field':0,'Bat':1}}
df.replace(encode, inplace=True) 
col=df.columns

nan_value_count=df.isna().sum()
df=df.dropna()  
df=df.drop(['1st innings win ratio'], axis=1)
df=df.drop(['venue'], axis=1)
df=df.drop(['2nd innings win ratio'], axis=1)
#df = df[(np.abs(stats.zscore(df.select_dtypes(exclude='object'))) < 3).all(axis=1)]

X=df.iloc[:,0:7]
print(X.columns)
y=df.iloc[:,7]

idx,c=np.unique(y,return_counts=True)
fig, axes = plt.subplots(1, 2,figsize=(7,5))
sns.barplot(x=idx,y=c,ax=axes[0])

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)

X_sm,y_sm=sm.fit_resample(X,y)

idx,c=np.unique(y_sm,return_counts=True)

sns.barplot(x=idx,y=c,ax=axes[1])

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2,random_state=0)



model = RandomForestClassifier(max_depth=3,criterion="entropy",random_state=135)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


y_pred=model.predict([[2024,4,0,0,0,0.4595,0.5405]])
print(y_pred)