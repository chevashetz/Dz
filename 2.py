import numpy as np
import math as mt
from tabulate import tabulate

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
    if i == 1:
        for j in range(n_values.size):
            value_rect[i][j] = integral_rect_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
            value_trap[i][j] = integral_trap_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
            value_simp[i][j] = integral_simp_numpy(functions[i], 0.0, mt.pi/2, n_values[j])
    else:
        for j in range(n_values.size):
            value_rect[i][j] = integral_rect_numpy(functions[i], 0.0, 1, n_values[j])
            value_trap[i][j] = integral_trap_numpy(functions[i], 0.0, 1, n_values[j])
            value_simp[i][j] = integral_simp_numpy(functions[i], 0.0, 1, n_values[j])

    table_data = []
    for j in range(n_values.size):
        table_data.append([n_values[j], value_rect[i][j], value_trap[i][j], value_simp[i][j]])

    print(f"\nРезультаты для функции f_{i + 1}:")
    print(tabulate(table_data, headers=[int(n), "Метод прямоугольников", "Метод трапеций", "Метод Симпсона"], floatfmt=".6f"))

'''
plt.plot(n_values, np.abs(3.0-value_rect[0][:]))
plt.plot(n_values, 1/n_values**1.0, '--')
plt.title("Абсолютная погрешность f_1")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel("$\epsilon$")
plt.show()

plt.plot(n_values, np.abs(1.0-value_rect[1][:]))
plt.plot(n_values, 1/n_values**1.0, '--')
plt.title("Абсолютная погрешность f_2")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel("$\epsilon$")
plt.show()

plt.plot(n_values, np.abs(0.632121-value_rect[2][:]))
plt.plot(n_values, 1/n_values**1.0, '--')
plt.title("Абсолютная погрешность f_3")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel("$\epsilon$")
plt.show()


plt.plot(n_values, np.abs(0.785398-value_rect[3][:]))
plt.plot(n_values, 1/n_values**1.0, '--')
plt.title("Абсолютная погрешность f_4")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$n$")
plt.ylabel("$\epsilon$")
plt.show()

plt.plot((b-a)/n_values, np.abs(3.0-value_rect[0][:]), label="Метод прямоугольников")
plt.plot((b-a)/n_values, np.abs(3.0-value_trap[0][:]), label="Метод трапеций")
plt.plot((b-a)/n_values, np.abs(3.0-value_simp[0][:]), label="Метод Симпсона")
plt.title("Абсолютная погрешность")
plt.legend()
plt.show()

plt.plot((b-a)/n_values, np.abs(3.0-value_rect[0][:]), label="Метод прямоугольников")
plt.plot((b-a)/n_values, np.abs(3.0-value_trap[0][:]), label="Метод трапеций")
plt.plot((b-a)/n_values, ((b-a)/n_values)**2.0, '--')
plt.title("Абсолютная погрешность")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("$\Delta x$")
plt.ylabel("$\epsilon$")
plt.show()
'''
p_values = np.arange(2, 8)
n = 2**p_values

def f_xy(x,y):
    return x**2.0+6.0*x*y+y**2.0

def a_bound(x):
    return -(1.0-x**2.0)**0.5

def b_bound(x):
    return (1.0-x**2.0)**0.5

def integral_rect2(f,a,b,c,d,n):
    def g(x):
        def f_y(y):
            return f_xy(x, y)
        return integral_rect_numpy(f_y, a(x), b(x), n)
    return integral_rect_numpy(g, c, d, n)

def integral_simp2(f,a,b,c,d,n):
    def g(x):
        def f_y(y):
            return f_xy(x, y)
        return integral_simp_numpy(f_y, a(x), b(x), n)
    return integral_simp_numpy(g, c, d, n)

integral_rect_values = np.zeros(n.size)
integral_simp_values = np.zeros(n.size)

integral_rect_values = [integral_rect2(f_xy, a_bound, b_bound, -1.0, 1.0, n) for n in n]
integral_simp_values = [integral_simp2(f_xy, a_bound, b_bound, -1.0, 1.0, n) for n in n]

print("\n")
for i, n in enumerate(n_values):
    print(f"n = {n}: Метод прямоугольников = {integral_rect_values[i]:.6f}, Метод Симпсона = {integral_simp_values[i]:.6f}")

integral_simp2(f_xy,a_bound,b_bound,-1.0,1.0,2**6)
# Оценка погрешности
reference_value_rect = integral_rect_values[-1]
reference_value_simp = integral_simp_values[-1]

error_rect = np.abs(integral_rect_values - reference_value_rect)
error_simp = np.abs(integral_simp_values - reference_value_simp)

print("\n")
for i, n in enumerate(n_values):
    print(f"n = {n}: Погрешность метода прямоугольников = {error_rect[i]:.6f}, Погрешность метода Симпсона = {error_simp[i]:.6f}")
print("\n")

'''
# Построение графика
plt.figure(figsize=(10, 6))
plt.loglog(n, error_rect, 'o-', label='Прямоугольники')
plt.loglog(n, error_simp, 's-', label='Симпсон')
plt.xlabel('Количество интервалов n')
plt.ylabel('Погрешность')
plt.title('Зависимость погрешности от n')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
'''

a, b, H = 0.0, 1.0, 1.0 #инициализируем начальные парараметры
npoint1 = 100
#определеяем функцию для расчета f(x)
def func(x):
    return (1.0-x**2.0)**0.5
np.random.seed(1)
xi = np.random.uniform(a, b, npoint1)
yi = np.random.uniform(0.0, H, npoint1)

'''
#вывод графика функции и точек
plt.plot(xi,yi,'o',label='$-(x_i,y_i)$')
xspace = np.linspace(a,b,101)
plt.plot(xspace,func(xspace),label='$-\sqrt{1-x^2}$')
plt.legend()
plt.show()
'''

nin = yi[yi<=func(xi)].size
for i in range(npoint1):
    if yi[i] <= func(xi[i]):
        nin += 1

def MonteCarlo1(f,a,b,H,npoint):
    np.random.seed(0)
    xi = np.random.uniform(a, b, npoint)
    yi = np.random.uniform(0.0, H, npoint)
    nin = yi[yi <= func(xi)].size
    return nin/npoint*(b-a)*H

npoint = np.logspace(1, 4, 101, dtype=int)
value = np.zeros(npoint.size)
error = np.zeros(npoint.size)
for i in range(npoint.size):
    value[i] = 4.0*MonteCarlo1(func, 0.0, 1.0, 1.0, npoint[i])
    error[i] = np.pi - value[i]
'''
plt.plot(npoint, error)
plt.xscale('log')
plt.show()
'''

def MonteCarlo2(f,a,b,H,npoint):
    np.random.seed(0)
    xi = np.random.uniform(a, b, npoint)
    yi = f(xi)
    return np.sum(yi)*(b-a)/npoint

for i in range(npoint.size):
    value[i] = 4.0*MonteCarlo2(func, 0.0, 1.0, 1.0, npoint[i])
    error[i] = np.pi - value[i]
'''   
plt.plot(npoint, error)
plt.xscale('log')
plt.show()
'''

n_min = npoint[np.where(error < 0.01)[0][0]]
print('Испытаний требуется для погрещности меньше 0.01:', n_min)
print("\n")

def rho_uniform(x,y):
  return 1.0

def rho_intro(x,y):
  return 1.0+0.5*(x**2.0+y**2.0)

def inside_disk(x,y, a=1):
  r_outer = 4.0
  r_inner = 1.0

  if x**2.0+y**2.0 <= r_outer**2.0:
    if (x-a)**2.0+y**2.0 > r_inner**2.0:
      return True
  return False

def MonteCarlo_Integr(rho_func,npoints):
    M = 0.0
    x_centerm = 0.0
    y_centerm = 0.0
    I = 0.0

    for _ in range(npoints):
      x = np.random.uniform(-4.0,4.0)
      y = np.random.uniform(-4.0,4.0)
      if inside_disk(x, y):
        rho = rho_func(x, y)
        M += rho
        x_centerm += x*rho
        y_centerm += y*rho
        I += (x**2.0+y**2.0) * rho

    area = np.pi*(4**2.0-1.0)
    M *= area/npoints
    x_centerm *= area/(npoints*M)
    y_centerm *= area/(npoints*M)
    I *= area/npoints
    return M, x_centerm, y_centerm, I

npoints2 = 10000
Muni, x_centermuni, y_centermuni, Iuni = MonteCarlo_Integr(rho_uniform, npoints2)
Mint, x_centermint, y_centermunint, Iint = MonteCarlo_Integr(rho_intro, npoints2)
print("\n")
print("M1 = ",Muni, "\nX1 = ",x_centermuni, "\nY1 = ", y_centermuni, "\nМомент инерции 1 = ", Iuni, "\nM2 = ", Mint, "\nX2 = ", x_centermint, "\nY2 = ", y_centermunint, "\nМомент инерции 2 = ", Iint)
