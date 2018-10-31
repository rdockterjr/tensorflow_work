import numpy as np
import matplotlib.pyplot as plt

def distnp(x1, x2):
    dist = np.linalg.norm(np.subtract(x1, x2))
    return dist

def errornp(x1, x2):
    dist = np.sum(np.subtract(x1, x2))
    return dist

nn = 100
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


#k-means
k = 2

samples = Data.shape[0]
dim = Data.shape[1]
means = Data[np.random.choice(samples, k, replace=False), :]
last_means = Data[np.random.choice(samples, k, replace=False), :]

classes = np.ones((samples,1))

break_cnt = 0
max_epoch = 1000
for epoch in range(max_epoch):
    sum_mean = np.zeros([k,dim])
    count = np.zeros([k,1])
    #find current mean
    for i in range(samples):
        sample = Data[i,:]
        min_dist = 100000000.0
        min_idx = 0
        for m in range(k):
            dist = distnp(sample, means[m,:])
            if dist < min_dist:
                min_dist = dist
                min_idx = m
        #sum new means
        sum_mean[min_idx,:] = np.add(sum_mean[min_idx,:], sample)
        count[min_idx] = count[min_idx] + 1
        classes[i] = min_idx
    #resample mean
    for m in range(k):
        means[m,:] = sum_mean[m,:] / count[m]

    #check if classes arent changing
    if(distnp(means, last_means) < 0.000001 ):
        break_cnt += 1
    if(break_cnt > 5):
        print('Early Break', epoch)
        break
    last_means = means
    
print(means)

#compute error
error = errornp(classes, Labels) / samples
if(error > 0.5):
    error = 1.0 - error
print(error)

# plot it
plt.scatter(d1[:,0], d1[:,1], label='C1', color='b', s=25, marker="o")
plt.scatter(d2[:,0], d2[:,1], label='C2', color='r', s=25, marker="o")
plt.scatter(means[0,0], means[0,1], label='M1', color='y', s=25, marker="x")
plt.scatter(means[1,0], means[1,1], label='M2', color='k', s=25, marker="x")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test')
plt.legend()
plt.axis((0,10,0,10))
plt.show()
