from flask import Flask, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

app = Flask(__name__)

df = pd.read_csv("ipl_data.csv")

# Preprocessing steps
df.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Delhi Capitals',
            'Chennai Super Kings','Punjab Kings','Rajasthan Royals', 
            'Gujarat Titans', 'Sunrisers Hyderabad', 'Lucknow Super Giants'],
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
df = df.dropna() 
df=df.drop(['venue','1st innings win ratio','2nd innings win ratio'],axis=1)

# SMOTE oversampling
X = df.iloc[:, 0:7]
y = df.iloc[:, 7]
sm = SMOTE(random_state=2)
X_sm, y_sm = sm.fit_resample(X, y)

# Train the model
model = RandomForestClassifier(max_depth=3, criterion="entropy", random_state=135)
model.fit(X_sm, y_sm)


@app.route('/')
def index():
    # Define a list of team numbers and their corresponding names
    team_numbers = [(0, 'MI'), (1, 'KKR'), (2, 'RCB'), (3, 'DC'), (4, 'CSK'),
                    (5, 'PBKS'), (6, 'RR'), (7, 'GT'), (8, 'SRH'), (9, 'LSG')]

    # Create options for team numbers dropdown
    team_options = ''.join([f'<option value="{num}">{name}</option>' for num, name in team_numbers])

    return f"""
   <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IPL Predictor</title>
        <style>
            body {{
               text-align: center;
               font-family: Arial, sans-serif;
               background-image: url('https://source.unsplash.com/1600x900/?cricket');
               background-size: cover;
               background-repeat: no-repeat;
               background-attachment: fixed;
               color: black;
            }}
            form {{
                margin: auto;
                width: 50%;
                border: 2px solid #ccc;
                padding: 20px;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.7);
            }}
            
            label, select, input[type="submit"] {{
                display: block;
                margin: 10px auto; /* Center the elements horizontally */
            }}
            input[type="submit"] {{
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }}
        </style>
        

        
    </head>
    <body>
        
        <form action="/" method="post">
            <h1>IPL Fever: Who Will Triumph?</h1>
            <label for="year">Year:</label>
            <input type="number" id="year" name="year" required><br><br>
            <label for="team1">Team 1:</label>
            <select id="team1" name="team1" required>
                {team_options}
            </select><br><br>
            <label for="team2">Team 2:</label>
            <select id="team2" name="team2" required>
                {team_options}
            </select><br><br>
            <label for="toss_winner">Toss Winner:</label>
            <select id="toss_winner" name="toss_winner" required>
                {team_options}
            </select><br><br>
            <label for="toss_decision">Toss Decision:</label>
            <select id="toss_decision" name="toss_decision" required>
                <option value="0">Field</option>
                <option value="1">Bat</option>
            </select><br><br>
        
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>

    """
    


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        team1 = int(request.form['team1'])
        team2 = int(request.form['team2'])
        toss_winner = int(request.form['toss_winner'])
        toss_decision = int(request.form['toss_decision'])

        # Fetch head-to-head ratio from the filtered DataFrames
        if not((df['team1'] == team1) & (df['team2'] == team2)).empty:
            head_to_head_team1 = df[(df['team1'] == team1) & (df['team2'] == team2)]['head to head ratio team1'].values[0]
            head_to_head_team2 = df[(df['team1'] == team1) & (df['team2'] == team2)]['head to head ratio team2'].values[0]
        if not((df['team1'] == team2) & (df['team2'] == team1)).empty:
            head_to_head_team1 = df[(df['team1'] == team1) & (df['team2'] == team2)]['head to head ratio team2'].values[0]
            head_to_head_team2 = df[(df['team1'] == team1) & (df['team2'] == team2)]['head to head ratio team1'].values[0]

        # Create input data for prediction
        input_data = [[year, team1, team2, toss_winner, toss_decision,
                       head_to_head_team1, head_to_head_team2]]

        # Predict the winner
        y_pred = model.predict(input_data)
        winner = 'Team 1' if y_pred[0] == team1 else 'Team 2'

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IPL Predictor</title>
            <style>
                body {{
                    text-align: center;
                    font-family: Arial, sans-serif;
                    background-image: url(' Dharam.jpg');
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    color: white;
                }}
                form {{
                    margin: auto;
                    width: 50%;
                    border: 2px solid #ccc;
                    padding: 20px;
                    border-radius: 10px;
                    background-color: rgba(255, 255, 255, 0.7);
                }}
                h1 {{
                    color: #000000;
                }}
                label, select, input[type="submit"] {{
                    display: block;
                    margin: 10px auto; /* Center the elements horizontally */
                }}
                input[type="submit"] {{
                    background-color: #4CAF50; /* Green */
                    border: none;
                    color: black;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <form>
                <h1>IPL Fever: Who Will Triumph?</h1>
                <h1>Predicted Winner for this match: {winner}</h1>
            </form>
        </body>
        </html>
        """

if __name__ == '__main__':
    app.run(debug=True)
