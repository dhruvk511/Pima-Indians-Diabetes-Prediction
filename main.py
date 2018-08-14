import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from classifiers import MLP


def load_data():
    directory = './data/'
    file = 'diabetes.csv'
    diabetes = pd.read_csv(directory+file)
    X = diabetes.iloc[:,0:-1].values
    y = diabetes.iloc[:,-1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, 
                                                                         test_size=0.2, random_state=4)
    return X_train, X_test, y_train, y_test

def main():
    """     layer_dims = [32, 12, 1]
    model = MLP(layer_dims)
    print(model)
    X_train, X_test, y_train, y_test = load_data()
    model.train_model(X_train,y_train)
    model.save_model() """
    
    dir = './models/2018-08-14_07:10:32'
    model = MLP.load_model(dir)
    print(model)
    

if __name__ == "__main__" : main()
