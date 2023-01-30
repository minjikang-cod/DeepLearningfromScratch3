import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator # 1. 함수를 가져온다.
        if f is not None: #창조 함수가 있으면
            x = f.input # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)  # 3. 함수의 역전파 알고리즘 실행
            x.backward() # 재귀로 하나 앞 변수의 역전파 시행

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 생성자함수 설정
        self.input = input # 입력 저장
        self.output = output # 출력 저장
        
        return output
        
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
        
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b 
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a 
assert y.creator.input.creator.input.creator == A 
assert y.creator.input.creator.input.creator.input == x 

####

y.grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)

####

y.grad = np.array(1.0)
y.backward() 
print(x.grad)