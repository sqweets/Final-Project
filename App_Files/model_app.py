import pickle
import flask
from flask import request
import numpy as np
from numpy import array

app = flask.Flask(__name__)

#getting our trained model from a file we created earlier
svm_model = pickle.load(open("Models/svm_model.pkl","rb"))

@app.route('/predict', methods=['POST'])
def predict():
    # grabbing a set of job features from the request's body
    new_features1 = []
    new_features2 = []
    feature_array = request.form
    feature_dict = dict(feature_array)

    # Getting the values from the dict and creating
    # 2 separate arrays
    index = 0
    for key, value in feature_dict.items():
        val_int = int(value[0])

        if (index == 0):
            new_features1.append(val_int)
            index = 1
        else:
            new_features2.append(val_int)
            index = 0

    # Combine the 2 arrays into a numpy array
    data = []
    data.append(new_features1)
    data.append(new_features2)
    sample = array(data)
    print(sample)

    # Our model predicts if treatment is sought
    prediction_svm = svm_model.predict(sample)

    # Sending our response object back as json
    return flask.jsonify(prediction_svm.tolist())

if __name__ == '__main__':
    app.run(debug=True)
