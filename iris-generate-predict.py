import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Advertising Prediction App")
st.write("This app predicts the **Number of Sales** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 1.0, 7.0, 4.0)
    Radio = st.sidebar.slider('Radio', 1.0, 7.0, 4.0)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 7.0, 4.0)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('Advertising')
X = data.drop(['sales'],axis=1)
Y = data.sales.copy()

modelGaussianAdvertising = GaussianNB()
modelGaussianAdvertising.fit(X, Y)

prediction = modelGaussianAdvertising.predict(df)
prediction_proba = modelGaussianAdvertising.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.advertisingtype())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
