import joblib
import numpy as np
import pandas as pd
import os
from preprocessing import preprocess
import math

from flask import Flask, request, redirect, render_template, jsonify
app = Flask(__name__)
import pickle
model = pickle.load(open('model.pkl','rb'))


columns = ['name', 'item_condition_id', 'brand_name', 'category_name', 'shipping', 'item_description']

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    # print(features)
    data = np.array(features)
    df = pd.DataFrame([data], columns=columns)
    processed_data = preprocess(df)
    pred = model.predict(processed_data)
    # prediction = np.expm1(scaler.inverse_transform(pred.reshape(-1, 1))[:,0])
    output = round(pred[0], 1)
    output=math.exp(output)
    return render_template('index.html', price='Recommended Price : ${:.2f}'.format(output))
    # return render_template('index.html', prediction_text='Predicted Price is: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))