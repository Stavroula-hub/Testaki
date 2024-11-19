import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Read input data from file shoppingData.csv into a pandas dataframe.
customerPurchases = pd.read_csv("shoppingData.csv", header=0, sep=',')

# Filter columns; keep only 'Age' and 'Purchase Amount (USD)' from loaded data
customerPurchases = customerPurchases[['Age', 'Purchase Amount (USD)']]

# Execute DBSCAN algorithm with eps equal to 2.27 and minimum number of samples
# equal to 3
dbscan = DBSCAN(eps=2.27, min_samples=3)
dbscan.fit(customerPurchases)

# Scatter plot of purchase data: Age on the x axis and purchase amount (USD)
# on the y axis. Points with same color indicate that they belong to the
# same cluster.
plt.scatter(x=customerPurchases['Age'], y=customerPurchases['Purchase Amount (USD)'], c=dbscan.labels_)

# Add title and x, y labels
plt.title('Purchases')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.show()

# Getting unique cluster labels
clusterLabels = np.unique(dbscan.labels_[dbscan.labels_>= 0])
print('Displaying', len(clusterLabels), 'clusters:')

# Iterating over cluster labels and filtering data based on cluster
# it belongs to.
# Displaying mean values for Age and Purchase Amount (USD) for each cluster
for lbl in clusterLabels:
    print('Cluster', lbl)
    print(customerPurchases[dbscan.labels_==lbl])
    print('Mean value of Age and Purchase amount (USD) for cluster', lbl)
    print(customerPurchases[dbscan.labels_==lbl].mean())
    print('')

print('Displaying noise points:')
print(customerPurchases[dbscan.labels_==-1])