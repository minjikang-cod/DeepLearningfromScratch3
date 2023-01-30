class Variable:
    def __init__(self, data):
        self.data = data 
        
##
import numpy as np

data = np.array(1.0)
x = Variable(data)

print(x.data)

##
x.data = np.array(2.0)

print(x.data)

##
x = np.array(1) # 스칼라 
print(x.ndim)

x = np.array([1,2,3])
print(x.ndim)

x = np.array([[1,2,3],[4,5,6]])
print(x.ndim)