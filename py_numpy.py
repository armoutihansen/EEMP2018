"""
###############################################################################
INTRODUCTION TO NUMPY
###############################################################################
##################################
Content:
    1. NumPy ndarray
    2. Universal functions
    3. Linear algebra
    4. Pseudorandom number generation
##################################
"""

print(__doc__)

#-----------------------------------------------------------------------------#


"""
NumPy is the fundamental package for scientific computing in Python. 
It is a Python library that provides a multidimensional array object, 
various derived objects (such as masked arrays and matrices), and 
an assortment of routines for fast operations on arrays, including
mathematical, logical, shape manipulation, sorting, selecting, 
I/O, discrete Fourier transforms, basic linear algebra, 
basic statistical operations, random simulation and much more.
"""

# 1. NumPy ndarray

# By convention, this is how we import numpy:
import numpy as np

"""
One of the key features of NumPy is its N-dimensional array object, or ndarray,
which is a fast, flexible container for large datasets in Python. Arrays enable you to
perform mathematical operations on whole blocks of data using similar syntax to the
equivalent operations between scalar elements.
"""
# Problem with lists:
height = [1.73, 1.68, 1.71, 1.86]
weight = [55, 48, 60, 85]

bmi = weight / (height ** 2)

# One solution:
bmi = [weight[i] / height[i] ** 2 for i in range(len(height))]

# Inefficient for large lists/arrays
we = np.array(weight)
he = np.array(height)

bmi2 = we / he ** 2

# draw an ndarray from the standard normal distribution
data = np.random.randn(2, 3)

#mathematical operations with data:

data * 10

data + data
data * 2

# Every array has a shape, a tuple indicating the size of each dimension

data.shape

# creating ndarrays:

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

# Arithmetic
arr2 * arr2
arr2 - arr2
1 / arr2
arr2 ** 0.5

arr = np.array([[2, 0, 4, 5], [4, 8, 3, 9]])
arr2 > arr

# Basic indexing and slicing
arr = np.arange(10)
arr[5]
# replace
arr[5] = 7

arr[5:8]
arr[5:8] = 12

arr_slice = arr[5:8]

# Boolean indexing
mask = arr2 > 4
arr2[mask]
arr3 = arr2[arr2 % 2 == 0]

# Transposing
arr = np.arange(32).reshape(8, 4)
arr.T

arr2 = np.arange(3, 35).reshape(4, 8)

# Dot product
arr@arr2


#-----------------------------------------------------------------------------#

# 2. Universal functions

# Fast Element-Wise Array Functions

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)

# Mathematical and statistical methods
arr = np.random.randn(5, 4)
np.mean(arr)
np.mean(arr, axis=0) # column-wise
np.mean(arr, axis=1) # row-wise

np.cumsum(arr, axis=0)
np.cumsum(arr, axis=1)

#-----------------------------------------------------------------------------#

# 3. Linear algebra

from numpy.linalg import inv
# Computing the dot product of X with its transpose X.T
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)

# as before, we could also do
inv(X.T@X)

#-----------------------------------------------------------------------------#

# 4. Pseudorandom number generation

# You can change NumPy’s random number generation seed using np.random.seed:
np.random.seed(1234)
"""
The data generation functions in numpy.random use a global random seed. To avoid
global state, you can use numpy.random.RandomState to create a random number
generator isolated from other
"""
rng = np.random.RandomState(999)
rng.randn(10)
np.random.randn(10)