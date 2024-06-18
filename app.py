import streamlit as st
import pickle
import pandas as pd

# Lists of IPL teams and cities where matches are held
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

# Load the trained machine learning model from a file
pipe = pickle.load(open('pipe.pkl','rb'))

# Set the title of the Streamlit app
st.title('IPL Win Predictor')

# Create two columns for user input
col1, col2 = st.columns(2)

# User selects the batting team from a dropdown list
with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))

# User selects the bowling team from a dropdown list
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

# User selects the host city from a dropdown list
selected_city = st.selectbox('Select host city',sorted(cities))

# User inputs the target score
target = st.number_input('Target')

# Create three columns for additional user inputs
col3,col4,col5 = st.columns(3)

# User inputs the current score
with col3:
    score = st.number_input('Score')

# User inputs the number of overs completed
with col4:
    overs = st.number_input('Overs completed')

# User inputs the number of wickets lost
with col5:
    wickets = st.number_input('Wickets out')

# Predict the probability when the button is clicked
if st.button('Predict Probability'):
    # Calculate the runs left to win
    runs_left = target - score

    # Calculate the balls left to play (20 overs total in IPL, so 120 balls)
    balls_left = 120 - (overs*6)

    # Calculate the remaining wickets (10 wickets in total)
    wickets = 10 - wickets

    # Calculate the current run rate
    crr = score/overs

    # Calculate the required run rate
    rrr = (runs_left*6)/balls_left

    # Create a DataFrame with the input features for prediction
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    # Get the prediction probabilities from the model
    result = pipe.predict_proba(input_df)

    # Probability of losing
    loss = result[0][0]

    # Probability of winning
    win = result[0][1]
    # Display the probabilities
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")