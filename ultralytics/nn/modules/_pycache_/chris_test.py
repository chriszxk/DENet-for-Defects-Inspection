import math

import numpy as np
import matplotlib.pyplot as plt
# import math
#
x = np.arange(-8, 8, 0.2)
# y=math.exp(x)
# plt.plot(x,y)
# plt.show()
from sympy import symbols, sin, plot
import  sympy
def func(y, x):
    return y**x

f=func(sympy.E, x)
# plt.plot(x, f)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


# 利用 numpy 实现 Sigmoid 函数
def sigmoid(x):
    # x 为输入值
    # y = sigmoid(x)
    y = 1 / (1 + np.exp(-x))
    # dy=y*(1-y)  # 若要实现 Sigmod() 的导数图像，打开此处注释，并返回 dy 值即可。
    return y


# 利用 matplotlib 来进行画图
def plot_sigmoid():
    # param:起点，终点，间距
    # x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    # f = func(sympy.E, -x)
    f=1-x
    # f=x**(2)
    # f=math.log(x)
    y1 = f*y
    # y2=x*y*2+y-y**2-x*(y**2)
    # plt.plot(x, y, c="r")
    # y2=-math.exp(-x)*sigmoid(x)+math.exp(-x)*sigmoid(x)(1-sigmoid(x))
    # y2=2*x*sigmoid(x)+(x**(2))*sigmoid(x)*(1-sigmoid(x))
    # y3=sigmoid(x)+x*sigmoid(x)*(1-sigmoid(x))
    # plt.plot(x, f, c="r")
    plt.plot(x, y1, c="r")
    # plt.plot(x, y2, c="b")
    # plt.plot(x, y3, c="y")
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()