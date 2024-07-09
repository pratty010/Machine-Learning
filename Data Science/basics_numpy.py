import numpy as np

## Define basic np arrays

# define one-dimensional arrays with different `dtype`
arr1 = np.array([1, 2, 3, 4], dtype=float)
arr2 = np.array([5, 6, 7, 8], dtype=int)

print(arr1[0] , arr1[0].dtype)
print(arr2[1] , arr2[1].dtype)

# define 3D arrays with nested list.
arr3 = np.array(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ]
    ])

print(arr3.shape) # print the shape of the array
print(arr3.dtype) # print the dtype of the array
print(arr3[0][1][1]) # access first 3*3 array --> access second row --> access second column
print(arr3.ndim) # print the number of rows


# create a multidimensional array with given shape and specified values
arr4 = np.zeros((2, 2, 2), dtype=int) # create array with given shape and all values as 0
arr5 = np.ones((3, 3, 3), dtype=int) # create array with given shape and all values as 1
arr6 = np.full((3, 4, 5), 10) # create multidimensional array with given shape and all values as 10

print(arr4)
print(arr5.size) # total elements of multidimensional array
print(arr5)
print(arr6)

# create an empty multidimensional array with given shape
arr7 = np.empty((3, 3, 3), dtype=int) # create array with given shape filled with undefined values
print(arr7)

# create a 2D array filled with random numbers
arr8 = np.random.rand(5, 5) # create array with given shape filled with random numbers between 0 and 1
print(arr8)

## automatically generate arrays with specified shape

# generate a list with start stop and steps
arr9 = np.arange(0, 1000, 2)
print(arr9)

# generate array with specified number of elements
arr10 = np.linspace(0, 1000, 101, dtype=int) 
print(arr10)

## Mathematical operations

# mathematical operations on arrays --> not implemented as common list operations
y1 = arr1 + arr2
y2 = arr1 * arr2
y3 = arr1 ** 2 + arr1

print(y1)
print(y2)
print(y3)

# vectorized mathematical operations on arrays 
y4 = np.add(arr1, arr2) # add 2 vector arrays
y5 = np.multiply(arr1, arr2) # multiply 2 vector arrays
y6 = np.power(arr1, 2) + np.power(arr1, 2) # array ^ 2
y7 = np.sin(arr1) + np.cos(arr1) # sin and cos of array elements
y8 = np.sqrt(arr2) # square root of array elements
y9 = np.exp(arr2) # e^x dor x in array elements
y10 = np.log(arr3) # log(x) for x in array elements

print(y4)
print(y5)
print(y6)
print(y7)
print(y8)
print(y9)
print(y10)

# analytical functions
y11 = np.sum(arr3) # sum of all elements in array
y12 = np.prod(arr3) # product of all elements in array
y13 = np.mean(arr3) # mean of all elements in array
y14 = np.std(arr3) # standard deviation of all elements in array
y15 = np.var(arr3) # variance of all elements in array
y16 = np.percentile(arr3, 50) # percentile of all elements in array
y17 = np.max(arr3) # maximum of all elements in array
y18 = np.min(arr3) # minimum of all elements in array
y19 = np.median(arr3) # median of all elements in array

print(y11)
print(y12)
print(y13)
print(y14)
print(y15)
print(y16)
print(y17)
print(y18)
print(y19)


## Reshaping and rearranging arrays

arr11 = np.array([
    [1, 2, 3, 4], 
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    ])

# reshape array to a new shape
y20 = arr11.reshape((3,2,2))
print(y20)

# transpose of arrays
y21 = arr11.T
print(y21)

# flattened arrays
y22 = y21.flat
for i in y22:
    print(i)

print(arr11.flatten())

## Joining arrays

arr12 = np.full((4, 6, 2), 10)
arr13 = np.full((4, 6, 2), -1)

# concatenate arrays along a specified axis
y23 = np.concatenate((arr12, arr13), axis=0)
print(y23)

# stack arrays along a specified axis
y24 = np.stack((arr12, arr13), axis=3)
print(y24)

y25 = np.hstack((arr12, arr13))
print(y25)

y26 = np.vstack((arr12, arr13))
print(y26)


## splitting the array

arr14 = np.array([
    [1, 2, 3, 4, 5, 6], 
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
    ])

# split array into multiple sub-arrays along a specified axis
y27 = np.split(arr14, 3, axis=0)
for i in y27:
    print(i)

y28 = np.split(arr14, 3, axis=1)
for i in y28:
    print(i)