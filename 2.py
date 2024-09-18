import numpy as np
import math

a = int(input(""))
b = int(input(""))
n =  int(intput(""))

def f_1(x):
    return  (intput(""))

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

value_rect = np.zeros(n.size)
for j in range(n.size):
 value_trap[j] = integral_trap_numpy(f_1,0.0,1.0,n[j])
value_trap

plt.plot((b-a)/n,np.abs(3.0-value_rect),label="Метод прямоугольников")
plt.plot((b-a)/n,np.abs(3.0-value_trap),label="Метод трапеций")
plt.plot((b-a)/n,np.abs(3.0-value_simp),label="Метод Симпсона")
plt.title("Абсолютная погрешность")
plt.legend()
plt.xlabel("$\Delta x$")
plt.ylabel("$\epsilon$")
plt.show()

plt.plot((b-a)/n,np.abs(3.0-value_rect),'-b',label="Метод прямоугольников")
plt.plot((b-a)/n,np.abs(3.0-value_trap),'-g',label="Метод трапеций")
plt.plot((b-a)/n,((b-a)/n)**1.0,'--b',label="$ \Delta x$")
plt.plot((b-a)/n,((b-a)/n)**2.0,'--g',label="$ \Delta x ^2$")
plt.title("Абсолютная погрешность")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$\Delta x$")
plt.ylabel("$\epsilon$")
plt.show()

#вторая задача
def a_bound(x):
 return -(1.0-x**2.0)**0.5

def b_bound(x):
 return (1.0-x**2.0)**0.5

def integral_simp2(f,a,b,c,d,n):
    def g(x):
        def f_y(y):
            return f_xy(x,y)
        return integral_simp(f_y,a(x),b(x),n)
    return integral_simp(g, c, d, n)
#3ая задача

a, b, H = 0.0, 1.0, 1.0 #инициализируем начальные парараметры
npoint = 100
#определеяем функцию для расчета f(x)
def MonteCarlo1(f,a,b,H,npoint):
    np.random.seed(0)
    xi = np.random.uniform(a,b,npoint)
    yi = np.random.uniform(0.0,H,npoint)
    nin = yi[yi<=func(xi)].size
    return nin/npoint*(b-a)*H
npoint = np.logspace(1,4,101,dtype=int)
value = np.zeros(npoint.size)
error = np.zeros(npoint.size)
for i in range(npoint.size):
 value[i] = 4.0*MonteCarlo1(func,0.0,1.0,1.0,npoint[i])
 error[i] = np.pi - value[i]
plt.plot(npoint,np.abs(error))
plt.xscale('log')
plt.yscale('log')
plt.show()

def MonteCarlo2(f,a,b,H,npoint):
    np.random.seed(0)
    xi = np.random.uniform(a,b,npoint)
    yi = f(xi)
    return np.sum(yi)*(b-a)/npoint

npoint = np.logspace(1,4,101,dtype=int)
value = np.zeros(npoint.size)
error = np.zeros(npoint.size)
for i in range(npoint.size):
 value[i] = 4.0*MonteCarlo2(func,0.0,1.0,1.0,npoint[i])
 error[i] = np.pi - value[i]
plt.plot(npoint,np.abs(error))
plt.xscale('log')
plt.yscale('log')
plt.show()
