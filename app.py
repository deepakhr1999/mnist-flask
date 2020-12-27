import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
from PIL import Image
from imageio import imread, imwrite as imsave
from skimage.transform import resize as imresize
import numpy as np
import keras.models
import re
import base64
from io import BytesIO

import sys 
sys.path.append(os.path.abspath("./model"))
from load import *

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
    
    out = model.predict(x)
    
    response = np.array_str(np.argmax(out, axis=1))
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
