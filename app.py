import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify
import json
from PIL import Image

from skimage.transform import resize as imresize
import numpy as np

import re
import base64
from io import BytesIO

import sys 
# sys.path.append(os.path.abspath("./model"))
from model.load import *

app = Flask(__name__)
global model
model = init()
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    x = parseImage(request.get_data())

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)
    
    preds = model.predict(x)
    
    response = predsToResponse(preds)
    
    return response 
    
def parseImage(imgData):
    """
        Reads the base64 image using PIL
        Resizes for Neural Net input
    """
    b64String = re.search(b'base64,(.*)', imgData).group(1)
    decoded = base64.b64decode(b64String)
    img = Image.open(BytesIO(decoded)).convert('L')
    x = np.invert(img)
    x = imresize(x,(28,28))
    return x

def predsToResponse(preds):
    """
        converts model predictions to json response
        expects preds = np.array of shape (1, 10)
    """
    preds = preds[0].round(3)
    top3  = (-preds).argsort()[:3].astype('int')
    top3 = [int(x) for x in top3]
    probs = [f"{preds[i]:.2f}" for i in top3]
    data = dict(
        top3  = top3,
        probs = probs,
        pred  = top3[0]
    )

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
