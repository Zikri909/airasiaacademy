import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# ADVERTISING")
st.write("This app predicts the **SALES** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 4.3, 7.9, 5.4)
    RADIO = st.sidebar.slider('RADIO', 2.0, 4.4, 3.4)
    NEWSPAPER = st.sidebar.slider('NEWSPAPER', 1.0, 6.9, 1.3)
   
    data = {'TV': TV,
            'RADIO': RADIO,
            'NEWSPAPER':NEWSPAPER,
           }
    features = pd.DataFrame(data, index=[0])
    return features
zul = user_input_features()



