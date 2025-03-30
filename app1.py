# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

filename = Path("model.sv")
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

df = pd.read_csv("DSP_1.csv")

min_age, max_age = float(df["Age"].min()), float(df["Age"].max())
min_sibsp, max_sibsp = int(df["SibSp"].min()), int(df["SibSp"].max())
min_parch, max_parch = int(df["Parch"].min()), int(df["Parch"].max())
min_fare, max_fare = int(df["Fare"].min()), int(df["Fare"].max())
# ustalanie zakresów

sex_d = {0:"Kobieta", 1:"Mężczyzna"}
pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

    st.set_page_config(page_title="Predykcja przeżycia katastrofy Titanica")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

    with overview:
        st.title("Czy przeżyłbyś katastrofę Titanica?")

    with left:
        pclass_radio = st.radio("Klasa pasażerska", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )
    
    with right:
        age_slider = st.slider("Wiek", value=min_age, min_value=min_age, max_value=max_age)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=min_sibsp, max_value=max_sibsp)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=min_parch, max_value=max_parch)
        fare_slider = st.slider("Cena biletu", min_value=min_fare, max_value=max_fare, step=1)
    
    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)
    
    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
