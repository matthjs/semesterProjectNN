import tensorflow as tf

# initialization of Tensors
x1 = tf.constant([[1,2,3], [4,5,6]])  # a 2x3 matrix
x2 = tf.ones((3,3))  # 3x3 matrix of ones
x3 = tf.zeros((2,3)) # 2x3 matrix of zeros
x4 = tf.eye(3) # I for the identity matrix (eye)
x5 = tf.random.normal((3,3), mean=0, stddev=1) # a 3x3 matrix where each element is from a normal distribution with mean 0 stddev 1
x6 = tf.random.uniform((1,3), minval=0, maxval=1) # a 1x3 matrix (i.e. vector in R^3) where each element is from a uniform distribution
x7 = tf.range(9) # a vector in R^9 that has elements 0,1, ...., 8
x8 = tf.range(start=1, limit=10, delta=2) # a vector 1,3,5,7,9
x9 = tf.cast(x8, dtype=tf.float64) # convert vector elements to data type float64

# Mathemetical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
z1 = tf.add(x, y) # vector addition x + y (z = x + y is equivalent)
z2 = tf.subtract(x, y) # vector addition x + (-y) <=> x - y (z = x - y is equivalent)
z3 = tf.divide(x, y) # element wise division (equivalent: x / y)
z4 = tf.multiply(x, y) # element wise multiplication (equivalent x * y)

d = tf.tensordot(x, y, axes=1) # dot product
# ... and many more
m1 = tf.random.normal((2, 3)) # random 2x3 matrix
m2 = tf.random.normal((3, 4)) # random 3x4 matrix
m3 = tf.matmul(m1, m2) # matrix multiplication
print(m3)


# Indexing

# Reshaping
x = tf.range(9)
x = tf.reshape(x, (3, 3)) # reshape vector in R^3 to matrix in R^{3x3}
x = tf.transpose(x, perm=[1,0]) # transpose

print(x)
