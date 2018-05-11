from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import numpy as np 
from sklearn.preprocessing import StandardScaler

import pickle

class MLPClassifier:
    """
    A multilayer perceptron classifer 
    The hidden layers of network comprise of rectified linear units. 
    The output layers is a single neuron with the sigmoid activation function.
    """

    def __init__(self):
        self.model = None
        self.scaler =None

   

    def train_model(self, layer_dims):
        """
        This function trains a multilayer perceptron model on the data and saves the model in the models directory

        Arguments :
            - layer_dims : A list of the number of neuron in each hidden layer except the output layer
        """
        #np.random.seed(7)

        dataset = np.loadtxt("./data/data_file.csv", delimiter=',')
        X = dataset[:,0:8]
        Y = dataset[:,8]
        # Normalizing the data with l2 normalization 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scaler = scaler

        # Defining the network architecture for the model 
        model = Sequential()
            # Adding dropout regularization layer
        model.add( Dropout(0.2, input_shape=(8,) ) )
            # Adding fully connected relu neuron layers 
        for neurons in layer_dims :
            model.add( Dense( neurons, activation='relu') )
            # Adding output layer 
        model.add( Dense( 1 , activation='sigmoid' ) )

        # Compiling the network to work with:
        #   Loss function : Cross Entropy 
        #   Optimization algorithm : Adam
        #sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
        model.fit(X,Y, epochs=150, verbose=0)
        
        # Evaluating accuracy of the model 
        scores = model.evaluate(X,Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        self.model = model
        
        # Serializing model to JSON
        model_json = model.to_json()
        with open("./models/MLP.json", "w") as json_file:
            json_file.write(model_json)
        # serializing weights to HDF5
        model.save_weights("./models/MLP.h5")
        # serializing the Normalizer 
        with open("./models/scaler.pkl","wb") as output_file:
            pickle.dump(scaler, output_file)
        print("Saved model to disk")



    def predict(self, X):
        """
        This function predicts the labels of the given inputs 
        If the model instance is not populated the saved model is loaded from the models directory
        Arguments :
            -  X : input parameters 
        Returns : 
            - predictions : the predictions computed by the MLP classifier
        """
        if self.model == None :
            # load the serialized model from the json file in the models directory and create model
            with open('MLP.json', 'r') as json_file:
                loaded_model_json = json_file.read()
                self.model = model_from_json(loaded_model_json)
                self.model.load_weights("MLP.h5")
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print("Loaded model from disk")
            # loading the serialized scaler file in 
            with open("./models/scaler.pkl","rb") as pkl_file:
                self.scaler = pickle.load(pkl_file)
        
        X_norm = self.scaler.transform(X)
        predictions = self.model.predict(X_norm)
        return predictions
