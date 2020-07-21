

import math
fibarray = [0,1]
def fab(n):
    if n < 0:
        print("please enter positive number")
    elif n<=len(fibarray):
        return fibarray[n-1]
    else:
        fib_value = fab(n-1)+fab(n-2)
        fibarray.append(fib_value)
        return fib_value

num = fab(30)
#print(num)
#print(fibarray)

newval = []
def goldenratio(array):
    if len(array) < 2:
        print("please pass array more than two Elements")
    elif len(array) > 2:
        for i in range(len(array)):
            if i < len(array) -2:
                val = array[i] / array[i+1]
                newval.append(val)
            elif i >= len(array)-2:
                print(i)
                val = array[i-1] / array[i-2]
                newval.append(val)
    else:
        print("done")

goldenratio(fibarray)
print(len(newval))

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
plt1 = Axes3D(fig)

plt1.scatter(newval[:10], newval[:10], newval[:10], color="red")
plt1.scatter(newval[10:20], newval[10:20], newval[10:20], color="blue", s=10,)
plt1.scatter(newval[20:], newval[20:], newval[20:], color="green")
plt.show()