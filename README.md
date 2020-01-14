# kMean implementation
Algorithm parameters   
Eucledean distance has been used as distance metric between the centroid and the sample.  
Algorithm stops when the abs difference between the new centroids and old centroids become less than 0.001.  

Testing  
Algorithm has been applied to Iris dataset.  
The test run the algorithm with K=3 and use kNN in order to classify the centroid and  
output the type and number samples asigned to each of them.  

One output from the test is the follow:  
Number of samples classified as Iris-versicolor : 63  
Number of samples classified as Iris-setosa : 50  
Number of samples classified as Iris-virginica : 37  
