import numpy as np
import math
def rms(y1,y2):
    output = []
    for i in range(len(y1)):
        output.append((y1[i] - y2[i]) ** 2)
    n = 1/(len(y1))
    return ((n * sum(output))**0.5)
def square_sum(x):
    output = 0
    for i in range(len(x)):
        output += x[i] **2
    return  output
def cube_sum(x):
    output = 0
    for i in range(len(x)):
        output += x[i] **3
    return  output
def power_4_sum(x):
    output = 0
    for i in range(len(x)):
        output += x[i] **4
    return  output

def mult_sum(x,y):
    output = 0
    for i in range(len(x)):
        output += x[i] * y[i]
    return output
def mult_sum2(x,y):
    output = 0
    for i in range(len(x)):
        output += x[i] * x[i] * y[i]
    return output
def one(x,y):
    A = np.array([[len(x), sum(x)], [sum(x), square_sum(x)]])
    B = np.array([[sum(y)], [mult_sum(x,y)]])
    a0 = (np.linalg.inv(A) @ B)[0][0]
    a1 = (np.linalg.inv(A) @ B)[1][0]
    new = []
    for i in range(len(x)):
        new.append(a1 * x[i] + a0)
    #print(a0, " ", a1, " ")
    return rms(y,new)
def two(x,y):
    A = np.array([[len(x), sum(x),square_sum(x)], [sum(x), square_sum(x),cube_sum(x)],[square_sum(x),cube_sum(x),power_4_sum(x)]])
    B = np.array([[sum(y)], [mult_sum(x, y)],[mult_sum2(x,y)]])
    a0 = (np.linalg.inv(A) @ B)[0][0]
    a1 = (np.linalg.inv(A) @ B)[1][0]
    a2 = (np.linalg.inv(A) @ B)[2][0]
    new = []
    for i in range(len(x)):
        new.append(a1 * x[i] + a0 + a2 * (x[i] ** 2))
    #print(a0, " ",a1, " ",a2, " ",)
    return rms(y, new)
def three(x,y):
    x1 = []
    for i in range(len(x)):
        x1.append(1/x[i])
    A = np.array([[len(x1), sum(x1)], [sum(x1), square_sum(x1)]])
    B = np.array([[sum(y)], [mult_sum(x1,y)]])
    a0 = (np.linalg.inv(A) @ B)[0][0]
    a1 = (np.linalg.inv(A) @ B)[1][0]
    new = []
    for i in range(len(x)):
        new.append(a1 * x1[i] + a0)
    return rms(y,new)
def four(x,y):
    x1 = []
    for i in range(len(x)):
        x1.append(np.log(x[i]))
    A = np.array([[len(x1), sum(x1)], [sum(x1), square_sum(x1)]])
    B = np.array([[sum(y)], [mult_sum(x1,y)]])
    a0 = (np.linalg.inv(A) @ B)[0][0]
    a1 = (np.linalg.inv(A) @ B)[1][0]
    new = []
    for i in range(len(x)):
        new.append(a1 * x1[i] + a0)
    return rms(y,new)
def five(x,y):
    y1 = []
    for i in range(len(y)):
        y1.append(1/y[i])
    A = np.array([[len(x), sum(x)], [sum(x), square_sum(x)]])
    B = np.array([[sum(y1)], [mult_sum(x,y1)]])
    a0 = (np.linalg.inv(A) @ B)[0][0]
    a1 = (np.linalg.inv(A) @ B)[1][0]
    new = []
    for i in range(len(x)):
        new.append(1/(a1 * x[i] + a0))
    return rms(y,new)
x = [float(i) for i in input().split()]
y = [float(i) for i in input().split()]
out = []
out.append(one(x,y))
out.append(two(x,y))
out.append(three(x,y))
out.append(four(x,y))
out.append(five(x,y))
print(out)
print(out.index(min(out)) + 1," ",min(out))


