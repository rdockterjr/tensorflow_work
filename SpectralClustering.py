import numpy as np
import matplotlib.pyplot as plt
import math

#https://towardsdatascience.com/spectral-clustering-for-beginners-d08b7d25b4d8
#http://www.cvl.isy.liu.se:82/education/graduate/spectral-clustering/SC_course_part1.pdf
#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

def distnp(x1, x2):
    dist = np.linalg.norm(np.subtract(x1, x2))
    return dist

def errornp(x1, x2):
    dist = np.sum(np.subtract(x1, x2))
    return dist

def gaussianKernel(d,sigma):
    dist = math.exp(-d/(2*sigma*sigma))
    return dist

nn = 5
dim = 2

#data1
s1 = (0.7, 0.7)
m1 = (2.0, 2.0)
d1 = np.random.normal(m1, s1, (nn, dim))
l1 = np.ones((nn,1))*0

#data2
s2 = (0.8, 0.8)
m2 = (5.0, 5.0)
d2 = np.random.normal(m2, s2, (nn, dim))
l2 = np.ones((nn,1))*1


#combine dataset
Data = np.concatenate((d1, d2), axis=0)
Labels = np.concatenate((l1, l2), axis=0)


#find k nearest neighbors
samples = Data.shape[0]
dim = Data.shape[1]
knn = 3
sigma = 1.0

#For storing
Adjacency = np.zeros((samples,samples))
Affinity = np.zeros((samples,samples))

#pairwise comparison
for i in range(samples):
    current = Data[i,:]
    #store all pairwise distances
    distances = np.zeros((samples,1))
    for j in range(samples):
        test = Data[j,:]
        distances[j] = distnp(current,test)
    #get the closest neighbors
    best_id = np.argsort(distances, axis=0)
    best_dist = distances[best_id]
    #store in matrix, skip self [0]
    for k in range(1,knn+1):
        #force undirectedness using a mutual knn graph
        #edges of the graph
        idn = best_id[k]
        Adjacency[i,idn] = 1
        #relations of the graph
        aff = gaussianKernel(best_dist[k],sigma)
        Affinity[i,idn] = aff

#remove non-symmetric elements (mutual knn)
for i in range(Adjacency.shape[0]):
    for j in range(i,Adjacency.shape[0]):
        if(Adjacency[i,j] != Adjacency[j,i]):
            Adjacency[i,j] = 0
            Adjacency[j,i] = 0
            Affinity[i,j] = 0
            Affinity[j,i] = 0

#get degree
Degree = np.diag( np.count_nonzero(Adjacency, axis=0) )

print(Adjacency)
print(Affinity)
print(Degree)

#Laplacian Matrix
Laplacian = np.subtract(Degree, Affinity)
print(Laplacian)

# plot it
plt.scatter(d1[:,0], d1[:,1], label='C1', color='b', s=25, marker="o")
plt.scatter(d2[:,0], d2[:,1], label='C2', color='r', s=25, marker="o")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test')
plt.legend()
plt.axis((0,10,0,10))
plt.show()
