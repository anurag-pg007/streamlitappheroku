import numpy as np
import pickle


#loading the saved model
loaded_model = pickle.load(open('C:/Users/MK/Desktop/Diabetesdeploy/trained_diabetes_model.sav','rb'))

#making a predictive system
input_data=(10,168,74,0,0,38,0.537,34)

#change to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshaping the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


#standardize the input data
#std_data=scaler.transform(input_data_reshaped)
#print(std_data)

#prediction
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0:
  print('Non-Diabetic')
else:
  print("Diabetic")
