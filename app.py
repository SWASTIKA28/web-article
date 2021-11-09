import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

import joblib 
pipe_lr = joblib.load(open("emotion_classifier.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    textinput = request.form.values()
    # final_features = [np.array(int_features)]
    prediction = pipe_lr.predict(textinput)

    output = prediction[0]

    return render_template('result.html', prediction_text='{}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     #prediction = model.predict([np.array(list(data.values()))])

#     #output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)