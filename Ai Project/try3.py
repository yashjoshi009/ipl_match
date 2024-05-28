import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("ipl_data.csv")


print(df["winner"].value_counts())