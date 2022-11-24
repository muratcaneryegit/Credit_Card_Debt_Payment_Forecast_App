import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


convert={
    "Male":1,
    "Female":2,
    "Graduate":1,
    "College":2,
    "High School":3,
    "Other":4,
    "Unknown1":5,
    "Unknown2":6,
    "Married":1,
    "Single":2,
    "Unknown":3,
    "Paid":-1,
    "1 Month Delay":1,
    "2 Months Delay":2,
    "3 Months Delay":3,
    "4 Months Delay":4,
    "5 Months Delay":5,
    "6 Months Delay":6

}

creditcard=pd.read_csv('UCI_Credit_Card.csv')
creditcard=creditcard[["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","BILL_AMT1","BILL_AMT2","PAY_AMT1","PAY_AMT2","default.payment.next.month"]]
##st.dataframe(creditcard)

train=st.sidebar.number_input("Train Size",0.4,0.9,value=0.8,step=0.1)
###MODEL SELECTION
model=st.sidebar.selectbox("Please Choose the Model",["Random Forest","Decision Tree","Logistic Regression"])
if model=="Random Forest":
    tree_=st.sidebar.number_input(" Please Enter the Number of Trees",value=100)


st.subheader("Predict")


##INPUTS
limit=st.sidebar.number_input("Card Limit",5000,1000000,value=10000)
sex=st.sidebar.selectbox("Sex",["Male","Female"])
education=st.sidebar.selectbox("Education",["Graduate","College","High School","Other","Unknown1","Unknown2"])
marriage=st.sidebar.selectbox("Marital Status",["Married","Single","Unknown"])
age=st.sidebar.number_input("Age",18,100,value=28,step=1)
pay0=st.sidebar.selectbox("Current Payment",["Paid","1 Month Delay","2 Months Delay","3 Months Delay","4 Months Delay","5 Months Delay","6 Months Delay"])
pay2=st.sidebar.selectbox("Current Payment 2",["Paid","1 Month Delay","2 Months Delay","3 Months Delay","4 Months Delay","5 Months Delay","6 Months Delay"])
bill_amt1=st.sidebar.number_input("Amount of Debt 1",step=1)
bill_amt2=st.sidebar.number_input("Amount of Debt 2",step=1)
pay_amt1=st.sidebar.number_input("Payment Amount",step=1)
pay_amt2=st.sidebar.number_input("Payent Amount 2",step=1)

calculate=st.sidebar.button("Calculate")



y=creditcard[['default.payment.next.month']]
x=creditcard.drop("default.payment.next.month",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train,random_state=34)


if model=="Decision Tree":
    tree=DecisionTreeClassifier()
    model=tree.fit(x_train,y_train)
elif model=="Random Forest":
    forest=RandomForestClassifier(n_estimators=tree_)
    model=forest.fit(x_train,y_train)
else:
    reg=LogisticRegression(solver="liblinear")
    model=reg.fit(x_train,y_train)

score=model.score(x_test,y_test)
st.write("Model Score: ",score)

if calculate:
    sonuc=model.predict([[limit,convert[sex],convert[education],convert[marriage],age,convert[pay0],convert[pay2],
                    bill_amt1,bill_amt2,pay_amt1,pay_amt2]])
    sonuc=sonuc[0]
    if sonuc==0:
        st.write("Debt Will Be Paid")
    elif sonuc==1:
        st.write("Debt Will Not Be Paid")


