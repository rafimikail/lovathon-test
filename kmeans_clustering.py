#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
print ("Data shape {}".format(dataset.shape))

# Sample data
dataset.head()

# Data type of each feature 
dataset.dtypes

#we will find and remove the duplicate entries in the datset
dataset.drop_duplicates(inplace = True)
print ("Data shape after dropping duplicates {}".format(dataset.shape))

#Checking for any null entries column wise, We can see that there are 0 null entries
print (pd.DataFrame(dataset.isnull().sum()))   

plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'],color='blue')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.show()

# Feature data
X=dataset.iloc[:,[3,4]].values
print(X.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X= sc_X.fit_transform(X)

# Using the Elbow method to find the number of clusters

#n_clusters is no.of clusters in a range
#k-means++ is an random initialization methods for centriods to avoid random intialization trap,
#max_iter is max no of iterations defined when k-means is running
#n_init is no of times k-means will run with different initial centroids
from sklearn.cluster import KMeans
wcss=[] #With in cluster sum of squers(Inertia)
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++",max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("wcss")
plt.show()


# Applying k-mean to dataset
kmeans=KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)
# Predcit cluster labels
y_kmeans=kmeans.fit_predict(X) #kmeans.labels_
print(y_kmeans)
print ("Unique no of Clusters {}".format(np.unique(y_kmeans)))

# Visualising the clusters
plt.figure(figsize = (12, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Customer Type 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Customer Type 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Customer Type 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Customer Type 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Customer Type 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "orange", label = "Centroids")
plt.title("Clusters of clients")
plt.xlabel("Annual Incom (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()


cluster_assignments = kmeans.labels_
dataset['Cluster'] = cluster_assignments
dataset['Genre']= dataset['Genre'].map({'Male':0, 'Female':1})
dataset = dataset.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'})
pd.set_option('display.max_rows', dataset.shape[0]+1)
dataset.to_pickle('clustered_profiles.pkl')