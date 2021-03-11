import os, pickle, argparse, pandas as pd
from sklearn.linear_model import LogisticRegression
from time import time

'''For ignoring useless warnings'''
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--savefile', required=True, help='model savefile name')
args = parser.parse_args()


"""
Here, we load the train and test data from
the mnist module from the tensorflow.keras library
"""
print('-----------------Data-----------------')
print('Loading MNIST data...')
train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
X_train, y_train = train.values[:, :-1], train.values[:, -1]
X_test, y_test = test.values[:, :-1], test.values[:, -1]

# normalize images such that pixels are in range [0.0,1.0]
numTrain, numTest = len(X_train), len(X_test)
X_train, X_test = X_train / 255, X_test / 255

print(f'Number of training examples: {numTrain}')
print(f'Training examples shape :', X_train.shape)
print()
print(f'Number of testing examples: {numTest}')
print(f'Testing examples shape: ', X_test.shape)
print()


"""
Section 1: DECLARE your model here!
The class LogisticRegression has already been imported from
sklearn.linear_model module.
Declare a logistic regression object and store it
in the variable 'model'
"""
print('-----------------Model-----------------')
### 1. Your code starts here

### 1. You code ends here
print(model)
print()


"""
Section 2: TRAIN your model
Use the model.fit method and pass X_train and y_train as arguments
"""
print('Training model :')
print('(This takes around a minute, depending on your cpu speed)')
start_time = time()
### 2. Your code starts here

### 2. You code ends here
train_duration = time() - start_time
print(f'Success: Finished training model in {train_duration} seconds')
print()

"""
Section 3: TESTING model
We predict our answer on test data and compare our predictions with
the actual test labels. We then display our accuracy on the test data.
Use the model.predict method and pass X_test as argument.
Store the value in the variable test_predictions
"""
print('Testing model...', end=' ')
### 3. Your code starts here

### 3. Your code ends here

acc = (test_predictions == y_test).mean()
print(f'accuracy on unseen data: {acc*100}% !')
print()

"""
We save the model as a pickle file for later use.
The following lines convert the model into a pickle file,
a standard way of storing variables in python.
"""
with open(args.savefile, 'wb') as file:
    pickle.dump(model, file)
print('Saved Logistic Regression model at', os.path.abspath(args.savefile))