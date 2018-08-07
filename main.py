import sys

from classifiers import MLP


def main():
    layer_dims = [32, 12, 1]
    model = MLP(layer_dims)
    print(model)

if __name__ == "__main__" : main()
