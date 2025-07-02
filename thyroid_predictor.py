import streamlit as st
import pickle
dict1={}
with open (r'C:\Users\mail4\Documents\luminar\ML\exersise\thyroid_cancer\model.pkl','rb')as f:
    dict1=pickle.load(f)

le_dict1=dict1['label_encoders']
oh_dict1=dict1['onehot_encoders']
scaler=dict1['scaler']

st.title('Tyroid cancer Prediction')
age=st.number_input("Age",0,100)
gender=st.selectbox("Gender",['Female','Male'])
gender=le_dict1['Gender'].transform([gender])[0]
country=st.selectbox("Country",['India','China','Nigeria','Brazil','Russia','Japan','South Korea','UK','USA','Germany'])
country=oh_dict1['Country'].transform([[country]])
country=country.flatten()
ethnicity=st.selectbox("Ethnicity",['Caucasian','Asian','African','Hispanic','Middle Eastern'])
ethnicity=oh_dict1['Ethnicity'].transform([[ethnicity]])
ethnicity=ethnicity.flatten()
family=st.selectbox("Family History",["No","Yes"])
family=le_dict1['Family_History'].transform([family])[0]
radiation=st.selectbox("Radiation Exposure",["No","Yes"])
radiation=le_dict1['Radiation_Exposure'].transform([radiation])[0]
iodine=st.selectbox("Iodine Deficiency",["No","Yes"])
iodine=le_dict1['Iodine_Deficiency'].transform([iodine])[0]
smoking=st.selectbox("Smoking",["No","Yes"])
smoking=le_dict1['Smoking'].transform([smoking])[0]
obesity=st.selectbox("Obesity",["No","Yes"])
obesity=le_dict1['Obesity'].transform([obesity])[0]
diabetes=st.selectbox("Diabetes",["No","Yes"])
diabetes=le_dict1['Diabetes'].transform([diabetes])[0]
tsh=st.number_input('TSH Level')
t3=	st.number_input('T3 Level')
t4=st.number_input('T4 Level')
nodule=st.number_input('Nodule Size')

data=[[age,gender,family,radiation,iodine,smoking,obesity,diabetes,tsh,t3,t4,nodule]+list(country)+list(ethnicity)]

button=st.button("Submit")
if button:
    print(dict1)
    scaled=scaler.transform(data)
    res=dict1['model'].predict(scaled)[0]
    if res==0:
        st.success("The Chances of Tyroid Cancer is Low")
        st.balloons()
    else:
        st.success("The Chances of Tyroid Cancer is High")