import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='CNN or LR')
args = parser.parse_args()


if args.model == 'CNN':
    """For CNN"""
    from ConvNeuralNet import loadModel, reshapeBeforeInput, modelName
    model = loadModel()

elif args.model == 'LR':
    """For LogisticRegression"""
    from LogisticRegression import loadModel, reshapeBeforeInput, modelName
    model = loadModel()
    model.predict = model.predict_proba    
else:
    print("Expecting --model as CNN or LR")
    exit(0)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html", modelName=modelName)

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    x = parseImage(request.get_data())

    # reshape image appropriately for model
    x = reshapeBeforeInput(x)
    Image.fromarray((255*x).reshape((28, 28)).astype('uint8')).save('output.png')
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
    x = resize(x,(28,28))
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
