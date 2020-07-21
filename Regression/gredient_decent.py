from numpy import *
import numpy as np

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, :-1]
        y = points[i, -1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(x))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, :-1]
        y = points[i, -1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

# b, m = gradient_descent_runner(points, 0, 0, 0.01,10000)
# m = m.reshape(2, 1)
# # # #print(compute_error_for_line_given_points(b, m, points))
# print("b----------->>>>",b ,"m----------->>>>",m.reshape(2, 1))
# print("---------------", (m).shape)
# #print(points[:, (0, 1)])
# print((points[:, :-1].dot(m)) + min(b))