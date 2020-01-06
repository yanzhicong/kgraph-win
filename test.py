from numpy import random
import pykgraph
import pickle as pkl
from time import time
import numpy as np
# dataset = random.rand(1000000, 16)
# query = random.rand(1000, 16)

from scipy.spatial.distance import cdist

dataset = pkl.load(open('test_plabel.pkl', 'rb'))

index = pykgraph.KGraph(dataset, 'euclidean')  # another option is 'angular'
# index.build(iterations=100, delta=1e-4, recall=0.999)                        #
index.build()
index.save("index_file");
# load with index.load("index_file");


print('start search')

k = 20

start = time()
knn = index.search(dataset, K=k, withDistance=True)                       # this uses all CPU threads
end = time()
print(end - start)


print(knn[0].shape, knn[0].dtype, knn[1].shape, knn[1].dtype)

dists = list(cdist(dataset, dataset[0:1], metric='euclidean',).reshape([-1]))
dists.sort()

dists = np.array(dists[0:200])



print(knn[1][0])
print(dists)

data = dataset[knn[0][0]]
print(data.shape)
dists = cdist(data, dataset[0:1], metric='euclidean',).reshape([-1]).astype(np.float32)
print(dists)

# print(knn[0])
# print(knn[1])
# print(knn[100])

# knn = index.search(query, K=10, threads=1)            # one thread, slower
# knn = index.search(query, K=1000, P=100)              # search for 1000-nn, no need to recompute index.



