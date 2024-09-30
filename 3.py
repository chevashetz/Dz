import numpy as np

def rho_uniform(x,y):
    return np.ones_like(x)

def rho_intro(x, y):
    return 1.0 + (x ** 2 + y ** 2)

def inside_disk(x, y, a=1.0):
    r_outer = 4.0
    r_inner = 1.0
    mask_outer = x**2 + y**2 <= r_outer**2
    mask_inner = (x - a)**2 + y**2 > r_inner**2
    return np.logical_and(mask_outer, mask_inner)

def MonteCarlo_Integr(rho_func, npoints):
    x = np.random.uniform(-4.0, 4.0, npoints)
    y = np.random.uniform(-4.0, 4.0, npoints)

    rho = rho_func(x, y)

    mask = inside_disk(x, y)

    rho[~mask] = 0.0

    sum_rho = np.sum(rho)
    sum_xrho = np.sum(x * rho)
    sum_yrho = np.sum(y * rho)
    sum_I = np.sum((x**2 + y**2) * rho)

    area_sampling_domain = 64
    M = sum_rho * area_sampling_domain / npoints
    x_centerm = sum_xrho / sum_rho
    y_centerm = sum_yrho / sum_rho
    I = sum_I * area_sampling_domain / npoints

    return M, x_centerm, y_centerm, I

npoints = 10000

Muni, x_centermuni, y_centermuni, Iuni = MonteCarlo_Integr(rho_uniform, npoints)

Mint, x_centermint, y_centermint, Iint = MonteCarlo_Integr(rho_intro, npoints)

print(f"Uniform Density:")
print(f"Mass = {Muni}")
print(f"Center of Mass = ({x_centermuni}, {y_centermuni})")
print(f"Moment of Inertia = {Iuni}\n")

print(f"Non-uniform Density:")
print(f"Mass = {Mint}")
print(f"Center of Mass = ({x_centermint}, {y_centermint})")
print(f"Moment of Inertia = {Iint}")

def x0(n):
    np.random.seed(52)
    x0 = np.random.uniform(a, b, n)
    return x0

def func (n):
  x = x0(n)
  return np.sin(np.sum(x**2.0))

def stdo(g, a, b, m, n):
  f1 = 0
  f2 = 0
  for i in range(m):

    f1 += g[i]**2
    f2 += g[i]

  return np.sqrt(abs(f1/m - (f2/m)**2))

def MonteCarloCube(f,a,b,m,n):
  summ = 0
  for i in range(m):
    summ = summ+f[i]
  return ((b-a)**n)*summ/m

n = 4
a = 0
b = 1
eps = 0.001
m = 10

point = np.zeros(m)
for i in range(m):
    point[i] = func(n)

while ((stdo(point,a,b,m,n)/(m**0.5)) >= eps):
    m += 1
    point = np.zeros(m)
    for i in range(m):
      point[i] = func(n)

Integ = MonteCarloCube(point, a, b, m, n)
sko = stdo(point, a, b, m, n)/(m**0.5)

print('Число итераций n=', m)
print('Интеграл=', Integ)
print('СКО=', sko)