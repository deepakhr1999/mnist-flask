from .load import loadModel
reshapeBeforeInput = lambda x: x.reshape((1, -1))
modelName = 'Logistic Regression'