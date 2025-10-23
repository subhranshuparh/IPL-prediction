import streamlit as st
import pickle
import pandas as pd

# IPL teams and cities lists
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load trained pipeline (ensure ipl.pkl is correct and present)
pipe = pickle.load(open('ipl.pkl', 'rb'))

# Title
st.title('IPL Win Probability Predictor')

# Team selectors
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Host city selector
selected_city = st.selectbox('Select host city', sorted(cities))

# Target runs
target = st.number_input('Target', min_value=1)

# Score, overs and wickets
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, format="%.1f")
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets_out
    # Prevent division by zero
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(f"{batting_team} - {round(win * 100)}% Win Probability")
    st.header(f"{bowling_team} - {round(loss * 100)}% Win Probability")
