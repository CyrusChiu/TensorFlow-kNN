# TensorFlow-kNN
kNN (k nearest neighbor) implementation in TensorFlow.

### Feature:
- batch prediction.
- user-defined distance metric.
- easy to use as sklearn.

#### Requirement:
tensorflow v1.4.0

### Usage:
```python
>>> from tf_knn import KNeighborsClassifier
>>> X = np.array([[0], [1], [2], [3]])
>>> y = [0, 0, 1, 1]
>>> neigh = KNeighborsClassifier(n_neighbors=3)

{'n_neighbors': 3, 'metric': 'euclidean', 'batch_size': 128}

>>> neigh.fit(X, y) 
>>> print(neigh.predict(np.array([[1.1]])))

[0]

>>> print(neigh.predict_proba(np.array([[0.9]])))

[[ 0.66666667  0.33333333]]
```

### More
http://cyruschiu.github.io/2017/12/01/knn-with-tensorflow/

**Author:** *Cyrus Chiu*
