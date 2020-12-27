import os
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from time import time

'''For ignoring useless warnings'''
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--savefile', required=True, help='model savefile name')
args = parser.parse_args()

from tensorflow.keras.datasets import mnist

print('-----------------Data-----------------')
print('Loading MNIST data...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

numTrain = len(X_train)
X_train = X_train.reshape( (numTrain, -1) ) / 255
print(f'Number training examples: {numTrain}')
print(f'Training examples shape :', X_train.shape)
print()


numTest = len(X_test)
X_test  = X_test.reshape( (numTest, -1) ) / 255
print(f'Number testing examples: {numTest}')
print(f'Testing examples shape: ', X_test.shape)
print()

print('-----------------Model-----------------')
model = LogisticRegression()
print(model)
print()

print('Training model :')
print('(This takes around a minute)')
start_time = time()
model.fit(X_train, y_train)
train_duration = time() - start_time
print(f'Success: Finished training model in {train_duration} seconds')
print()

print('Testing model...', end=' ')
acc = model.score(X_test, y_test)
print(f'accuracy on unseen data: {acc*100}% !')
print()

with open(args.savefile, 'wb') as file:
    pickle.dump(model, file)
print('Saved Logistic Regression model at', os.path.abspath(args.savefile))