import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대 0으로 초기화
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대 기록 (부모 세대(= 생성자 함수 세대)+1) 
    
    def clear_grad(self):
        self.grad = None
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set() # 만난 적(계산한 적) 있는 함수 저장
        
        def add_funcs(f): # 메서드 안 내장 메서드
            if f not in seen_set: # 계산한 적 없으면 
                funcs.append(f)
                seen_set.add(f) 
                funcs.sort(key = lambda x : x.generation) # 세대 수로 정렬
        
        add_funcs(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    add_funcs(x.creator)

                    
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs]) # 입력 변수가 둘 이상일 때 가장 큰 generation 수 선택
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs 
        self.outputs = outputs
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y 
    
    def backward(self, gy):
        return gy, gy
    
class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx 
    
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward() 

print(y.data)
print(x.grad)