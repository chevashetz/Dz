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

npoints = 100000


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
