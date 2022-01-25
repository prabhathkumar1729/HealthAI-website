import flask, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template ,url_for, jsonify # Import flask libraries
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__,template_folder="templates")
#----------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html') # Render index.html

@app.route('/Home.html')
def home():
    return render_template('Home.html') # Render home.html

#----------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/About.html')
def about():
    return render_template('About.html') # Render About.html

#----------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/Contact.html')
def contact():
    return render_template('Contact.html') # Render Contact.html

#----------------------------------------------------------------------------------------------------------------------------------------------------

diabdataset = pd.read_csv('Datasets\\diabetes.csv')
diabdataset_X = diabdataset.iloc[:,[1, 2, 5, 7]].values
sc = MinMaxScaler(feature_range = (0,1))
diabdataset_scaled = sc.fit_transform(diabdataset_X)
diabmodel = pickle.load(open('diabetesmodel.pkl', 'rb'))

@app.route('/Diabetes-Prediction.html')
def diabetes():
    return render_template('Diabetes-Prediction.html') # Render Diabetes-prediction.html
@app.route('/predictdiabetes',methods=['POST'])
def predictdiabetes():
    final_features = [
[
int(request.form["Glucose Level"]),
int(request.form["Insulin"]),
float(request.form["BMI"]),
int(request.form["Age"])
]
]
    print(final_features)
    prediction = diabmodel.predict(sc.transform(final_features))
    if prediction == 1:
        output = "You have Diabetes, please consult a Doctor"
    elif prediction == 0:
        output = "You don't have Diabetes, you are safe"
    return render_template('Diabetes-Prediction.html', prediction_text=output)
#----------------------------------------------------------------------------------------------------------------------------------------------------

caloriesburntmodel = pickle.load(open('caloriesburntmodel.pkl', 'rb'))
@app.route('/Calories-Burnt-Prediction.html')
def calories():
    return render_template('Calories-Burnt-Prediction.html') # Render Calories-Burnt-Prediction.html

@app.route('/predictcaloriesburnt',methods=['POST'])
def predictcaloriesburnt():
    final_features = [
[
int(request.form["Gender"]),
int(request.form["Age"]),
int(request.form["Height"]),
int(request.form["Weight"]),
int(request.form["Duration"]),
int(request.form["Heartrate"]),
int(request.form["Temperature"])
]
    ]
    coloriesburnt = caloriesburntmodel.predict(final_features)
    return render_template('Calories-Burnt-Prediction.html', prediction_text="Total calories burnt: {}".format(coloriesburnt[0]))


#----------------------------------------------------------------------------------------------------------------------------------------------------

heartdiseasemodel = pickle.load(open('heartmodel.pkl', 'rb'))


@app.route('/Heart-Disease-Prediction.html')
def heart():
    return render_template('/Heart-Disease-Prediction.html') # Render Heart-Disease-Prediction.html

@app.route('/predictcoronaryheartdisease',methods=['POST'])
def predictcoronaryheartdisease():
    final_features = [
[
float(request.form["Age"]),
float(request.form["cholesterol level"]),
float(request.form["Systolic blood pressure"]),
float(request.form["Diastolic blood pressure"]),
float(request.form["BMI"]),
float(request.form["Heartrate"]),
float(request.form["Glucose level"])
]
    ]
    prediction = heartdiseasemodel.predict(final_features)
    if prediction == 1:
        return render_template('Heart-Disease-Prediction.html', prediction_text="You have coronary heart disease, please consult a Doctor")
    elif prediction == 0:
        return render_template('Heart-Disease-Prediction.html', prediction_text="You don't have coronary heart disease, you are safe")

#----------------------------------------------------------------------------------------------------------------------------------------------------

breastcancermodel = pickle.load(open('breastcancermodel.pkl', 'rb'))
breastcancerscaler = pickle.load(open('breastcancerscaler.pkl', 'rb'))

@app.route('/Breast-Cancer-Prediction.html')
def cancer():
    return render_template('/Breast-Cancer-Prediction.html') # Render Breast-Cancer-Prediction.html

@app.route('/predictbreastcancer',methods=['POST'])
def predictbreastcancer():
    final_features = [
[
float(request.form["Texture Mean"]),
float(request.form["Area Mean"]),
float(request.form["Concavity Mean"]),
float(request.form["Area SE"]),
float(request.form["Concavity SE"]),
float(request.form["Fractal Dimension SE"]),
float(request.form["Smoothness Worst"]),
float(request.form["Concavity Worst"]),
float(request.form["Symmetry Worst"]),
float(request.form["Fractal Dimension Worst"])
]
]
    print(final_features)
    final_features = breastcancerscaler.transform(final_features)    
    prediction = breastcancermodel.predict(final_features)
    y_probabilities_test = breastcancermodel.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('Breast-Cancer-Prediction.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    else:
         return render_template('Breast-Cancer-Prediction.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))

#----------------------------------------------------------------------------------------------------------------------------------------------------

if(__name__=='__main__'):
    app.run(debug=True)