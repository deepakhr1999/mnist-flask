from .load import loadModel
modelName = 'Convolutional Neural Network'
reshapeBeforeInput = lambda x: x.reshape(1, 28, 28, 1)