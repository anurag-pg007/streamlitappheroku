import pickle
import numpy as np
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open(r'C:\Users\MK\Desktop\Diabetesdeploy\trained_diabetes_model.sav','rb'))

#function for prediction
def diabetes_prediction(input_data):
    # making a predictive system
    #input_data = (10, 168, 74, 0, 0, 38, 0.537, 34)

    # change to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data
    # std_data=scaler.transform(input_data_reshaped)
    # print(std_data)

    # prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The Person is Non-Diabetic'
    else:
        return "The Person is Diabetic"

def main():
    #giving a title
    st.title('Diabetes Prediction Web App')

    #getting the input from user

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Level')
    SkinThickness = st.text_input('SkinThickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('  BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age')


    #code for prediction
    diagonosis=''

    #creating a button
    if st.button('Diabetes Test Result'):
        diagonosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

        st.success(diagonosis)

if __name__ == '__main__':
    main()
