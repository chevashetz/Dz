import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import math

a = 0
b = 1.0
n = 5
def f_1(x):
    return 2.0*x+3.0*x**2.0+4.0*x**3.0

def f_2(x):
    return np.cos(x)

def f_3(x):
    return np.exp(-x)

def f_4(x):
    return (1.0+x**2.0)**(-1.0)

def integral_rect_numpy(f,a,b,n):
    x = np.linspace(a,b,n+1)
    s = 0
    s = np.sum(f(x[1:n]))
    dx = (b-a)/n
    s = s*dx
    return s

def integral_trap_numpy(f,a,b,n):
    x = np.linspace(a,b,n+1)
    s = 0
    s = np.sum(f(x[1:n]))
    s = s+0.5*f(x[0])
    s = s+0.5*f(x[n])
    dx = (b-a)/n
    s = s*dx
    return s

def integral_simp_numpy(f,a,b,n):
    x = np.linspace(a,b,n+1)
    s = 0
    s = 4.0*np.sum(f(x[1:n:2]))
    s = s + 2.0*np.sum(f(x[2:n:2]))
    s = s + f(x[0])
    s = s + f(x[n])
    dx = (b-a)/n
    s = s*dx/3.0
    return s

n = np.array([2,4,8,64,128,1024])

value_trap= np.zeros(n.size)
for j in range(n.size):
 value_trap[j] = integral_trap_numpy(f_1,0.0,1.0, n[j])
 print(value_trap)
print("\n")

value_trap= np.zeros(n.size)
for j in range(n.size):
 value_trap[j] = integral_trap_numpy(f_2,0.0,1.0, n[j])
 print(value_trap)
print("\n")

value_trap= np.zeros(n.size)
for j in range(n.size):
 value_trap[j] = integral_trap_numpy(f_3,0.0,1.0, n[j])
 print(value_trap)
print("\n")

value_trap= np.zeros(n.size)
for j in range(n.size):
 value_trap[j] = integral_trap_numpy(f_4,0.0,1.0, n[j])
 print(value_trap)

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

