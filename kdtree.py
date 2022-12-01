import numpy as np
from sklearnex import patch_sklearn
import time
import sys
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier


num_thread = 72

if len(sys.argv) == 2:
    num_thread = int(sys.argv[1])

def set_daal_num_threads(num_threads):
    try:
        import daal4py
        if num_threads:
            daal4py.daalinit(nthreads=num_threads)
    except ImportError:
        raise('@ Package "daal4py" was not found. Number of threads '
                     'is being ignored')
train_file_x = "bigdata/synthetic-classification-10-X-train-250000x16_256_tile.npy"
train_file_y = "bigdata/synthetic-classification-10-y-train-250000x16_256_tile.npy"

train_file_x = "synthetic-classification-10-X-train-16000000x16.npy"
train_file_y = "synthetic-classification-10-y-train-16000000x16.npy"

# train_file_x = "bigdata/synthetic-classification-10-X-train-250000x16_8_tile.npy"
# train_file_y = "bigdata/synthetic-classification-10-y-train-250000x16_8_tile.npy"

# train_file_x = "bigdata/synthetic-classification-10-X-train-2000000x16_8_tile_col.npy"
# train_file_y = "bigdata/synthetic-classification-10-y-train-2000000x16_8_tile_col.npy"

# train_file_x = "bigdata/synthetic-classification-10-X-train-16000000x16_8_tile_col.npy"
# train_file_y = "bigdata/synthetic-classification-10-y-train-16000000x16_8_tile_col.npy"

train_file_x = "data/synthetic-classification-10-X-train-250000x16.npy"
train_file_y = "data/synthetic-classification-10-y-train-250000x16.npy"
test_file_x  = "data/synthetic-classification-10-X-train-250000x16.npy"
test_file_y  = "data/synthetic-classification-10-y-train-250000x16.npy"
X_test = np.load(test_file_x)
y_test = np.load(test_file_y)

data_order = "F"
set_daal_num_threads(num_thread)

X_train = np.load(train_file_x)
y_train = np.load(train_file_y)
print(type(X_train), np.isfortran(X_train), isinstance(X_train, np.ndarray))
print(type(y_train))
if data_order == 'F':
    X_train = np.asfortranarray(X_train, float)
    y_train = np.asfortranarray(y_train, float)
    print(type(X_train), np.isfortran(X_train), isinstance(X_train, np.ndarray))
    print(type(y_train), isinstance(np.asarray([1]), np.ndarray))

elif data_order == 'C':
    X_train = np.ascontiguousarray(X_train, float)
    y_train = np.ascontiguousarray(y_train, float)

knn_clsf = KNeighborsClassifier(n_neighbors=10,
                                weights='uniform',
                                algorithm='kd_tree',
                                metric='euclidean')
print(y_train, y_train.shape)
# exit()
start = time.time()
print("start")
print(start)
print("...")
# X_train = np.asfortranarray(X_train, float)
knn_clsf.fit(X_train, y_train)

print("duration")
print((time.time() - start)*1000000)
names = train_file_x.partition("-train-")
print(names[0])
print(names[-1])
# print(knn_clsf.predict(X_train),'\n', y_train)
print("accuracy")
y_pred = knn_clsf.predict(X_test)
print((y_test == y_pred).sum()/250000)

print(y_test)
print(y_pred)