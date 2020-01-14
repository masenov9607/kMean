from kMean import kMean
from scipy.io import arff
import pandas as pd
from itertools import count
import sys
from KNN import KNN

def prepare_dataset(input_file):
    data = arff.loadarff(input_file)
    dataset = pd.DataFrame(data[0])
    return dataset[dataset.columns[:-1]]

def convert_to_iris_flower(sample):
    input_file = "iris.arff"
    predictor = KNN(input_file)
    predictor.normalize()
    flower = predictor.classify_sample(sample)
    return flower



def main():
    input_file = "iris.arff"
    dataset =prepare_dataset(input_file)
    sample = dataset.iloc[0]
    predictor = kMean(dataset)
    predictor.assign_centroid()
    predictor.Run()
    predictions = predictor.dataset.groupby(["centroid"]).count()
    num = predictions[predictions.columns[0]].values
    i = count()
    for sample in predictor.centroids:
        flower = convert_to_iris_flower(sample)
        n = num[next(i)]
        print(f"Number of samples classified as {flower} : {n}")



if __name__ == "__main__":
    main()
