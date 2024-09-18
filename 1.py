import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

def f(x):
    return x**2 + 2 * x**3

def J(t):
    return 10 * np.exp(-5 * t)

x = np.linspace(-100.0, 100.0, 200)
t = np.linspace(-100.0, 100.0, 200)

y = f(x)
z = J(t)

ax1 = plt.subplot(gs[0])
ax1.plot(x, y, 'r-')
ax1.set_title('График f(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid()

ax2 = plt.subplot(gs[1])
ax2.plot(t, z, 'b-')
ax2.set_title('График J(t)')
ax2.set_xlabel('t')
ax2.set_ylabel('J(t)')
ax2.grid()

m = int(input("Введите значение для m: "))

a = np.linspace(1.0, 10.0, 10)
b = np.zeros(10)

for i in range(10):
    b[i] = a[i]**3

print("a:", a)
print("b:", b)


def f_sum(v, m):
    sum = np.zeros_like(v)
    for n in range(1, m):
        sum += (-1)**n * np.sin(n * v) / n
    return sum


def y(v, m):
    sum = f_sum(v, m)
    return 1 - sum

v = np.linspace(0, 2 * np.pi, 100)
w = y(v, m)

ax3 = plt.subplot(gs[2])
ax3.plot(v, w, 'g-')
ax3.set_title('График y(x)')
ax3.set_xlabel('x')
ax3.set_ylabel('y(x)')
ax3.grid()

plt.tight_layout()
plt.show()
