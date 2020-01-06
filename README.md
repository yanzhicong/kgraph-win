
This repo is modified to work under Windows with Python3.7 and Visual Studio 2015(vc140)


# Note : Build Boost on windows machine

It's a really tough work, for the pre-build installer on Boost website are built with python2 and do not support python3, I need to rebuild it from the source. I summary the right steps in the following:

1. Download boost 1.64.0 source code zip and unzip into any position(please make sure is the right version).
2. Modify the following file: .\libs\python\src\converter\builtin_converters.cpp(51)

from 
```
      return PyUnicode_Check(obj) ? _PyUnicode_AsString(obj) : 0;
```
to
```
      return PyUnicode_Check(obj) ? (void*)_PyUnicode_AsString(obj) : (void*)0;
```
3. Create user-config.jam under boost source code directory, and add the python version and path to this file. For me, I write the following code to the jam file:
```
using python 
    : 3.7 
    : C:/Anaconda3/python.exe 
    : C:/Anaconda3/include
    : C:/Anaconda3/libs
    ;
using python 
    : 2.7 
    : C:/Anaconda3/envs/py27/python.exe 
    : C:/Anaconda3/envs/py27/include
    : C:/Anaconda3/envs/py27/libs
    ;
using python 
    : 3.6 
    : C:/Anaconda3/envs/py36/python.exe 
    : C:/Anaconda3/envs/py36/include
    : C:/Anaconda3/envs/py36/libs
    ;
```
4. Run bootstrap.bat
5. Build and install shared library:
```
.\b2.exe --prefix=C:/Boost --user-config=./user-config.jam address-model=64 install link=shared 
```
6. Build and install static library:
```
.\b2.exe --prefix=C:/Boost --user-config=./user-config.jam address-model=64 install link=static 
```

KGraph: A Library for Approximate Nearest Neighbor Search
=========================================================

# Introduction

KGraph is a library for k-nearest neighbor (k-NN) graph construction and
online k-NN search using a k-NN Graph as index.  KGraph implements 
heuristic algorithms that are extremely generic and fast:
* KGraph works on abstract objects.  The only assumption it makes is
that a similarity score can be computed on any pair of objects, with
a user-provided function.
* KGraph is among the fastest of libraries for k-NN search according to [recent benchmark](https://github.com/erikbern/ann-benchmarks).

For best generality, the C++ API should be used.  A python wrapper
is provided under the module name pykgraph, which supports Euclidean
and Angular distances on rows of NumPy matrices.


# Building and Installation

KGraph depends on a recent version of GCC with C++11 support, cmake
and the Boost library.  The package can be built and installed with
```sh
cmake -DCMAKE_BUILD_TYPE=release .
make
sudo make install
```

A Makefile.plain is also provided in case cmake is not available.

The Python API can be installed with
```
python setup.py install
```

# Python Quick Start

```python
from numpy import random
import pykgraph

dataset = random.rand(1000000, 16)
query = random.rand(1000, 16)

index = pykgraph.KGraph(dataset, 'euclidean')  # another option is 'angular'
index.build(reverse=-1)                        #
index.save("index_file");
# load with index.load("index_file");

knn = index.search(query, K=10)                       # this uses all CPU threads
knn = index.search(query, K=10, threads=1)            # one thread, slower
knn = index.search(query, K=1000, P=100)              # search for 1000-nn, no need to recompute index.
```

Both index.build and index.search supports a number of optional keywords
arguments to fine tune the performance.  The default values should work
reasonably well for many datasets.  One exception is that reverse=-1 should be
added if the purpose of building index is to speedup search, which is the
typical case, rather than to obtain the k-NN graph itself.

Two precautions should be taken:
* Although matrices of both float32 and float64 are supported, the latter is not optimized.  It is recommened that
matrices be converted to float32 before being passed into kgraph.
* The dimension (columns of matrices) should be a multiple of 4.  If not, zeros must be padded.

For performance considerations, the Python API does not support user-defined similarity function,
as the callback function is invoked in such a high frequency that, if written in Python, speedup will
inevitably be brought down.  For the full generality, the C++ API should be used.

# C++ Quick Start

The KGraph C++ API is based on two central concepts: the index oracle and the search oracle.
(Oracle is a fancy way of calling a user-defined callback function that behaves like a black box.)
KGraph works solely with object IDs from 0 to N-1, and relies on the oracles to map the IDs to
actual data objects and to compute the similarity. To use KGraph, the user has to extend the following
two abstract classes

```cpp
    class IndexOracle {
    public:
        // returns size N of dataset
        virtual unsigned size () const = 0;
        // computes similarity of object 0 <= i and j < N
        virtual float operator () (unsigned i, unsigned j) const = 0;
    };

    class SearchOracle {
    public:
        /// Returns the size N of the dataset.
        virtual unsigned size () const = 0;
	/// Computes similarity of query and object 0 <= i < N.
        virtual float operator () (unsigned i) const = 0;
    };
```

The similarity values computed by the oracles must satisfy the following two conditions:
* The more similar the objects are, the smaller the similarity value (0.1 < 10, -10 < 1).
* Similarity must be symmetric, i.e. f(a, b) = f(b, a).

KGraph's heuristic algorithm does not make assumption about properties such as
triangle-inequality.  If the similarity is ill-defined, the worst it can do is to lower
the accuracy and to slow down computation.

With the oracle classes defined, index construction and online search become straightfoward:

```cpp
#include <kgraph.h>

KGraph *index = KGraph::create();

if (need_to_create_new_index) {
    MyIndexOracle oracle(...);	// subclass of kgraph::IndexOracle
    KGraph::IndexParams params;  
    params.reverse = -1;
    index->build(oracle, params);
    index->save("some_path");
}
else {
    index->load("some_path");
}

MySearchOracle oracle(...);	// subclass of kgraph::SearchOracle

KGraph::SearchParams params;
params.K = K;
vector<unsigned> knn(K);    	// to save K-NN ids.
index->search(oracle, params, &knn[0]);
// knn now contains the IDs of k-NNs, highest similarity in the front

delete index;
```

Note that the search API does not directly imply nearest neighbor search.  Rather
it is a generic API for minimizing a function on top of a graph, and finds the K
nodes where the function assumes minimal values.

# More Documentation
### Oracles for Common Tasks
KGraph provides a number of [efficient oracle implementation](doc/oracle.md) for
common tasks. 
### [Parameter Tuning](doc/params.md)
### [API Documentation from Old Web Site](http://www.kgraph.org/index.php?n=Main.API)
### [Doxygen Documentation](http://aaalgo.github.io/kgraph/doc/html/annotated.html)

http://www.kgraph.org/


