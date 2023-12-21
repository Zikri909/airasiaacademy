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

st.subheader('User Input parameters')
st.write(zul)

data = sns.load_dataset('ADVERTISING')
X = data.drop(['SALES'],axis=1)
Y = data.SALES.copy()

modelGaussianSALES = GaussianNB()
modelGaussianSALES.fit(X, Y)

prediction = modelGaussianSALES.predict(zul)
prediction_proba = modelGaussianSALES.predict_proba(zul)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
