from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import adam
from keras.regularizers import l2

import numpy as np 
from sklearn.preprocessing import StandardScaler

import pickle
import json
import os
from datetime import datetime

class MLP:
    """
    A multilayer perceptron classifer 
    The hidden layers of network comprise of rectified linear units WITH L2 regularization.\n
    No dropout layers\n
    The output layers is a single neuron with the sigmoid activation function.\n
    The loss functino is set to 'binary_entropy'\n
    The optimizer is set to 'adam'
    """

    def __init__(self, layer_dims):
        self.model = None
        self.score = None
        self.scaler = None
        self.layer_dims = layer_dims
        # Default values for the hyperparameters 
        self.hyperparameters = { 'learning_rate' : 0.1 ,
                                 'decay' : 0.001 ,  
                                 'epochs' : 100 ,
                                 'batch_size' : 50 ,
                                 'beta1' : 0.9 ,
                                 'beta2' : 0.999 ,
                                 'epsilon' : 0.00000001 ,
                                 'regularization' : [ 0 for _ in range( len(layer_dims) ) ] }
        
    def __str__(self):
        return("This is a MLP Network object\n With no of Neurons in each layer = "+str(self.layer_dims)+
               "\n Hyperparameters "+str(self.hyperparameters))
    
    def printSomething(self):
        print("Something")
        
    def train_model( self, X, y ):
        """
        This function trains a multilayer perceptron model on the data 
        And saves the model in the models directory

        Arguments :-
            - X  : feature ndarray
            - y  : targets ndarray
        """
        #np.random.seed(7)

        # Defining the network architecture for the model 
        model = Sequential()
        n_features = X.shape[0]
        
        # Adding fully connected relu neuron layers 
        model.add( Dense( self.layer_dims[0], 
                          input_dim=n_features,
                          activation='relu',
                          kernel_regularizer=l2(self.hyperparameters['regularization'][0] ) 
                        ))
                       
        for l in range(1, len( self.layer_dims )) :   
            model.add( Dense( self.layer_dims[l], 
                              activation='relu',
                              kernel_regularizer=l2(self.hyperparameters['regularization'][l] )
                            ))          
        # Adding output layer 
        model.add( Dense( 1 , activation='sigmoid' ) ) 
        
        adam_optimizer = adam( lr = self.hyperparameters['learning_rate'] ,
                               beta_1 = self.hyperparameters['beta1'] ,
                               beta_2 = self.hyperparameters['beta2'] , 
                               epsilon = self.hyperparameters['epsilon'] , 
                               decay = self.hyperparameters['decay']  ) 
        
        model.compile( loss='binary_crossentropy', 
                       optimizer=adam_optimizer, 
                       metrics=['accuracy'] )
        
        model.fit( X, y, 
                   epochs =  self.hyperparameters['epochs'] ,  
                   batch_size = self.hyperparameters['batch_size'] , 
                   verbose = 0 ) 
        
        # Evaluating accuracy of the model 
        scores = model.evaluate( X, y )
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
        self.model = model
        
     
     
    def save_model(self):
        """
        The function saves the model and the weights into the direcotry named with the current 
        date and time in the "models" directory 
        """
        #creating a new directory to save the model 
        new_dir = "./models/" + str( datetime.now() ).split('.')[0].replace(' ','_') 
        if not os.path.exists(new_dir) :
            os.makedirs(new_dir)

        # Serializing model to JSON
        model = self.model
        model_json = model.to_json() 
        with open(new_dir+"/MLP.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(new_dir+"/MLP.h5") # serializing weights to HDF5
        print("Saved model to disk")


    def load_model(self, model_dir):
        """
        This function loads a pre-trained model for the given directory 
        """
        # load the serialized model from the json file in the models directory and create model
        with open(model_dir+'/MLP.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(model_dir+"/MLP.h5")
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("Loaded model from disk")   


    def predict(self, X, scaler):
        """
        This function predicts the labels of the given inputs 
        If the model instance is not populated the saved model is loaded from the models directory
        
        Arguments :-
            - X : input parameters 
        Returns :-
            - predictions : the predictions computed by the MLP classifier
        """
        if self.model != None :
            X_norm = self.scaler.transform(X)
            predictions = self.model.predict(X_norm)
            return predictions
        else :
            raise Exception("Please train the model or load a pre-trained model")
            

