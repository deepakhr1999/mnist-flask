import pickle

def loadModel(filename='SavedModels/logreg.pkl'):
    print("Opening logreg.pkl file that you saved using train.py")
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model