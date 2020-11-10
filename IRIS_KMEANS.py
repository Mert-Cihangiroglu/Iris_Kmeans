import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


iris = datasets.load_iris()
iris2 =iris.target_names
data = iris.data

samples_num, dim = data.shape  # I am finding and assigning  the number of samples and dimension of the samples
num_arr = np.arange(0, samples_num)
np.random.shuffle(num_arr)
centroids = data[num_arr[:3], :] # I am picking 3 centroids from the shuffled data and initial centroids.

samples_num = data.shape[0]
cluster_data = np.array(np.zeros((samples_num, 2)))

def eucl_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))    # takes two values and calculates the euclidian distance between them.

def k_means(data, k):
    samples_num = data.shape[0]
    cluster_data = np.array(np.zeros((samples_num, 2)))
    cluster_changed = True
    centroids = data[num_arr[:3], :]
    print("Initial centers：\n", centroids) 
    count = 0

    while cluster_changed:
        count += 1
        cluster_changed = False
        for i in range(samples_num):
            min_dist = 100000.0
            min_index = 0
            for j in range(k):
                distance = eucl_distance(centroids[j, :], data[i, :])  # Calculating the distances between centroids and the each of the row in our data set comparing with each centroid      
                if distance < min_dist:
                    min_dist = distance           # I am assigning the min_distance to the distance came from euclidian distance function
                                                  #then I am comparing the distance of the current row with each centroid 
                                                  # So if the current distance will be higher than the distance I will get from my euclidian function I will change the current cluster and assign it to the new one
                    cluster_data[i, 1] = min_dist # I am saving the current distance between my row and the centroid 
                    min_index = j                 #I am changing the cluster of the row with the J.(J is the current centroid's cluster)  
                    
            if cluster_data[i, 0] != min_index:   # On the last raw when I did not change the min_index(cluster of the raw) anymore I cluster_changed will stay as false and i will go out of my loop
                cluster_changed = True            #If I did change the min_index from the above if statement i will assign the last j(0,1 or 2 whatever it was left on) of the loop and will continue to 
                                                  #Iterate.
                cluster_data[i, 0] = min_index    # cluster data holds the cluster no and the distance to the centroid
            
        for j in range(k):  
            cluster_index = np.nonzero(cluster_data[:, 0] == j) # cluster indexes
            points_in_cluster = data[cluster_index] #now it is time to take the values of  points in the cluster  and we will calculate the new mean of each cluster 
            centroids[j, :] = np.mean(points_in_cluster,axis=0) # Calculate the mean of each clusters and give an updated version of the centers of each cluster
            
            

    print("Number of iterations：", count)         # I am counting the iterations I made in total
    return centroids, cluster_data                 #returning new centroids and also the cluster data which holds cluster no and the distance to the centroid

def calculate_accuracy(cluster_data, k_num): 
    right = 0
    for k in range(0, k_num):
        checker = [0, 0, 0]
        for i in range(0, 50):
            checker[int(cluster_data[i + 50 * k, 0])] += 1 # so in here we check first 50 second 50 and third 50 elemets of the cluster data . if it turns 0 we increase checker[0] by one
                                                           # we increase checker[1] if it the cluster data gives us 1 as a result and we increase checker[2] by one if it returns 2 to us.
                                                           # we take max value of the checker after each cluster so we know how many correct clustering our model has done        
        right += max(checker)
        
    return right
      


centroids, cluster_data = k_means(data, 3) 
print("cluster centers new：\n", centroids)
right_num = calculate_accuracy(cluster_data, 3) # Calling the accuracy function to give me the number of correct clusters
print("Error Rate:", 1 - right_num / len(data))  # calculating the error rate

predictions = []
for i in range(150):  
    predictions.append(int(cluster_data[i,0]))

data2 = pd.DataFrame(data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

plt.figure(figsize=(18,5))
colors = np.array(['red', 'green', 'blue'])
red_patch = mpatches.Patch(color='red', label=0)
green_patch = mpatches.Patch(color='green', label=1)
blue_patch = mpatches.Patch(color='blue', label=2)
plt.figure(figsize=(18,5))
plt.subplot(1, 2, 1)
plt.scatter(data2['Petal Length'], data2['Petal Width'], c = colors[predictions])
plt.title("Model's classification")
plt.legend(handles=[red_patch, green_patch, blue_patch])




