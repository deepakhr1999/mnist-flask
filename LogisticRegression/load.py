import pickle

def loadModel(filename='SavedModels/logreg.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model