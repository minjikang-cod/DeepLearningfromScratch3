class Variable:
    def __init__(self, data):
        self.data = data
        
class Function:
    def __call__(self, input):
        x = input.data # input은 Variable 인스턴스
        y = self.forward(x) # self를 붙여줘라!!!!!!
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2
    
##
x = Variable(10)
f = Function()
y = f(x)
print(type(y)) # type() 함수 : 객체의 클래스
print(y.data)

##
x = Variable(10)
f = Square()
y = f(x)
print(type(y))
print(y.data)
