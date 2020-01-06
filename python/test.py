#!/usr/bin/env python
import sys
import time
from numpy import random
import pykgraph
import pickle as pkl

N = 1000000
Q = 1000
D = 132
TYPE = 'f'
USE_SKLEARN = True

def eval (gold, result):
    result = result[:gold.shape[0],:]
    assert gold.shape == result.shape
    N = gold.shape[0]
    K = gold.shape[1]
    total = 0
    for i in range(N):
        total += len(set(gold[i]).intersection(result[i]))
    return 1.0 * total / (N * K)


dataset = pkl.load(open('./test_plabel.pkl', 'rb'))
query = pkl.load(open('./test_plabel.pkl', 'rb'))[:1000]


# dataset = random.rand(N, D).astype(TYPE)
# query = random.rand(Q, D).astype(TYPE)


index = pykgraph.KGraph(dataset, "euclidean")
K=10
#index.build(reverse=-1)
#index.save("index_file");
# load with index.load("index_file");

gold = None
if USE_SKLEARN:
    print("Generating gold standard...")
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(dataset)
    start = time.time()
    distances, gold = nbrs.kneighbors(query)
    print("Time:", time.time() - start)

print("Searching with brute force...")
start = time.time()
result = index.search(query, K=K)                        # this uses all CPU threads, set prune=1 to make index smaller (no accuracy loss)
bf_time = time.time() - start
print("Time:", bf_time)
if gold is None:
    gold = result
else:
    print("Recall:", eval(gold, result))

print("Searching with BLAS...")
start = time.time()
result = index.search(query, K=K, blas=True)
blas_time = time.time() - start
print("Time:", blas_time)
print("Recall:", eval(gold, result))

print("Building index...")
index.build(reverse=-1)
print("Searching with index...")
start = time.time()
result = index.search(query, K=K)
index_time = time.time() - start
print("Time:", index_time)
print("Recall:", eval(gold, result))

print("blas startup:", bf_time / blas_time)
print("index startup:", bf_time / index_time)
