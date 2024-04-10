# -*- coding: utf-8 -*-

#importing libraries
import joblib
import inputScript
import sklearn
#load the pickle file
classifier = joblib.load('final_models/svm_final2.pkl')

#input url
print("enter url")
url = input()

#checking and predicting
checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)

# print(prediction)

# x = prediction.tolist()
#print(type(prediction))

print(prediction)