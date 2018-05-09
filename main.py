import sys

from MLPClassifier import MLPClassifier

def main():
    model = MLPClassifier()
    layer_dims = [ 12, 8 ]
    model.train_model( layer_dims )

if __name__ == "__main__" : main()
