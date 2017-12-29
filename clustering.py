
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm

import pandas as pd
import numpy as np
 

df=pd.DataFrame()
df= (pd.read_csv('C:/Users/neeraj/Desktop/cluster-mockdata.csv',delimiter=','))

#df1=df["prevCo":"Performance"]


newdf = df[df.columns[1:3]]    # 1-3
numpyMatrix = newdf.as_matrix()
#print(numpyMatrix)


target = df[df.columns[9:]]
tagetMatrix = target.as_matrix()
#print(numpyMatrix)

#plt.subplot(1, 2, 1)
plt.scatter(numpyMatrix[:,0],numpyMatrix[:,1],s=150)     #  ,linewidth=2.5
plt.title('data points plot')


#plt.show()


clf=KMeans(n_clusters=2)
clf.fit(numpyMatrix)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","y."]


for i in range(len(numpyMatrix)):
    plt.plot(numpyMatrix[i][0],numpyMatrix[i][1],colors[labels[i]],markersize=10)

#plt.subplot(1, 2, 2)
plt.scatter(centroids[:,0],centroids[:,1],marker="X",s=150,linewidth=5)
plt.title('k-means with centroids')
plt.xlabel('Prev_Co')
plt.ylabel('GPA')

plt.show()



for i in range(len(numpyMatrix)):
    
    plt.subplot(1, 2, 1)
    plt.plot(numpyMatrix[i][0],numpyMatrix[i][1],colors[tagetMatrix[i]],markersize=10)
    plt.title('Real Classification')
    
    plt.subplot(1, 2, 2)
    plt.plot(numpyMatrix[i][0],numpyMatrix[i][1],colors[clf.labels_[i]],markersize=10)
    plt.title('K-Means Classification')


plt.show()


# The fix, we convert all the 1s to 0s and 0s to 1s.
predY = np.choose(clf.labels_, [1, 0, 2]).astype(np.int64)
'''
print (clf.labels_)
print (predY)
'''

acc=sm.accuracy_score(tagetMatrix, predY)
print("accuracy is ",acc)

conf=sm.confusion_matrix(tagetMatrix, predY)
print("conf is", conf)



