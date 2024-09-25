import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math as mt

a = 0
b = 1.0
n = 5

# Define functions
def f_1(x):
    return 2.0*x + 3.0*x**2.0 + 4.0*x**3.0

def f_2(x):
    return np.cos(x)

def f_3(x):
    return np.exp(-x)

def f_4(x):
    return (1.0 + x**2.0)**(-1.0)

# List of functions
functions = [f_1, f_2, f_3, f_4]

# Integral functions
def integral_rect_numpy(f, a, b, n):
    x = np.linspace(a, b, n+1)
    s = np.sum(f(x[1:n]))
    dx = (b-a)/n
    s = s * dx
    return s

def integral_trap_numpy(f, a, b, n):
    x = np.linspace(a, b, n+1)
    s = np.sum(f(x[1:n]))
    s = s + 0.5*f(x[0])
    s = s + 0.5*f(x[n])
    dx = (b-a)/n
    s = s * dx
    return s

def integral_simp_numpy(f, a, b, n):
    x = np.linspace(a, b, n+1)
    s = 4.0 * np.sum(f(x[1:n:2]))
    s = s + 2.0 * np.sum(f(x[2:n:2]))
    s = s + f(x[0])
    s = s + f(x[n])
    dx = (b-a)/n
    s = s * dx / 3.0
    return s


n_values = np.array([2, 4, 8, 64, 128, 1024])

value_rect = np.zeros((4, n_values.size))
value_trap = np.zeros((4, n_values.size))
value_simp = np.zeros((4, n_values.size))

for i in range(4):
    if i == 2:
        for j in range(n_values.size):
            value_rect[i][j] = integral_rect_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
            value_trap[i][j] = integral_trap_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
            value_simp[i][j] = integral_simp_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
    else:
        for j in range(n_values.size):
            value_rect[i][j] = integral_rect_numpy(functions[i], 0.0, 1, n_values[j])
            value_trap[i][j] = integral_trap_numpy(functions[i], 0.0, 1, n_values[j])
            value_simp[i][j] = integral_simp_numpy(functions[i], 0.0, 1, n_values[j])

plt.plot(n_values, np.abs(3.0-value_rect[0][:]))
plt.plot(n_values, 1/n_values**1.0, '.')
plt.title("Абсолютная погрешность f_1")
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(n_values, np.abs(1-value_rect[1][:]))
plt.plot(n_values, 1/n_values**1.0, '.')
plt.title("Абсолютная погрешность f_2")
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(n_values, np.abs(0.632121-value_rect[2][:]))
plt.plot(n_values, 1/n_values**1.0, '.')
plt.title("Абсолютная погрешность f_3")
plt.xscale("log")
plt.yscale("log")
plt.show()


plt.plot(n_values, np.abs(0.785398-value_rect[3][:]))
plt.plot(n_values, 1/n_values**1.0, '.')
plt.title("Абсолютная погрешность f_4")
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot((b-a)/n_values, np.abs(3.0-value_rect[0][:]), label="Метод прямоугольников")
plt.plot((b-a)/n_values, np.abs(3.0-value_trap[0][:]), label="Метод трапеций")
plt.plot((b-a)/n_values, np.abs(3.0-value_simp[0][:]), label="Метод Симпсона")
plt.title("Абсолютная погрешность")
plt.legend()
plt.show()




'''
plt.plot((b-a)/n_values, np.abs(3.0-value_rect[0][:]), label="Метод прямоугольников")
plt.plot((b-a)/n_values, np.abs(3.0-value_trap[0][:]), label="Метод трапеций")
plt.plot((b-a)/n_values, ((b-a)/n)**2.0, '--')
plt.title("Абсолютная погрешность")
plt.legend()
plt.show()
'''

'''
#plt.plot((b-a)/n,np.abs(3.0-value_rect),label="Метод прямоугольников")
plt.plot((b-a)/n,np.abs(3.0-value_trap),label="Метод трапеций")
#plt.plot((b-a)/n,np.abs(3.0-value_simp),label="Метод Симпсона")
plt.title("Абсолютная погрешность")
plt.legend()
plt.show()

#plt.plot((b-a)/n,np.abs(3.0-value_rect),'-b',label="Метод прямоугольников")
plt.plot((b-a)/n,np.abs(3.0-value_trap),'-g',label="Метод трапеций")
plt.title("Абсолютная погрешность")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
'''

