import numpy as np
np.random.seed(0)
from numpy.fft import ifft
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})


N = 10000
c0 = 2
mu = np.array([0, -200, 0])
sigmas = np.array([100, 50, 100])
cov = np.diag(sigmas)


def F_hat(omega):
    return omega**2 * np.exp(-omega**2)

def G_hat(omega, x, y):
    return 1 / (4*np.pi*norm(x-y)) * np.exp(1j * omega/c0*norm(x-y))

def plot_F_hat():
    omega = np.linspace(0,20,100)
    f_vals = F_hat(omega)
    plt.figure()
    plt.plot(omega, f_vals)
    plt.show()

def plot_ys(ys):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ys[:,0], ys[:,1], ys[:,2])
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    plt.savefig("q1/plot_ys.png")
    return


omega = np.linspace(-3,3,101)
ys = np.random.multivariate_normal(mu ,cov, N)
with open("q1/ys.npy", "wb") as f:
    np.save(f, ys)


def get_xs(case):
    if case == 1:
        xs = np.array([[0,50*j,0] for j in range(5)])
        return xs
    if case == 2:
        xs = np.array([[0,5*j,0] for j in range(5)])
        return xs
    else:
        xs = np.array([[50*(j-2),100,0] for j in range(5)])
        return xs


def C_N(tau, x1, x2):
    res = 0
    step = omega[1] - omega[0]
    for y in ys:
        f_val = F_hat(omega)
        g_val1 = G_hat(omega, x1, y)
        g_val2 = G_hat(omega, x2, y)
        assert f_val.shape == g_val1.shape
        assert f_val.shape == g_val2.shape
        int_val = np.sum(f_val*np.conj(g_val1)*g_val2*np.exp(-1j*omega*tau)*step)
        res += int_val
    res /= 2*np.pi*N
    return np.real(res)

def C_N_ifft(x1, x2):
    C_N_vals = np.zeros(omega.shape)
    for y in tqdm(ys):
        f_val = F_hat(omega)
        g_val1 = G_hat(omega, x1, y)
        g_val2 = G_hat(omega, x2, y)
        assert f_val.shape == g_val1.shape
        assert f_val.shape == g_val2.shape
        fourier_vals = f_val*np.conj(g_val1)*g_val2
        fourier_vals = np.roll(fourier_vals, -len(omega)//2)
        int_vals = np.real(ifft(fourier_vals))
        C_N_vals += int_vals
    C_N_vals /= 2*np.pi*N
    return np.real(C_N_vals)


xs = get_xs(case=2)
x1 = xs[1]
x2 = xs[0]
C_N_vals = C_N_ifft(x1, x2)
plt.figure()
plt.plot(C_N_vals)
plt.savefig("Essai_C_N_x2_x1.png")



# for case in [1, 2, 3]:
#     xs = get_xs(case)
#     taus = np.linspace(-10,10,100)
#     C_N_vals = [dict() for j in range(5)]
#     for j in range(5):
#         for tau in tqdm(taus):
#             C_N_vals[j][tau] = C_N(tau, xs[j], xs[0])
#         with open(f"q1/case{case}/C_N_x{j+1}.npy", "wb") as f:
#             np.save(f, C_N_vals[j])
#         plt.figure()
#         plt.plot(C_N_vals[j].keys(), C_N_vals[j].values())
#         plt.xlabel(r"$\tau$")
#         plt.ylabel(r"$C_N(\tau,x_{j+1},x_1)$")
#         plt.savefig(f"q1/case{case}/plot_x{j+1}x1.png")

# for case in [2]:
#     xs = get_xs(case)
#     taus = np.linspace(-10,10,100)
#     C_N_vals = [dict() for j in range(5)]
#     for j in range(5):
#         for tau in tqdm(taus):
#             C_N_vals[j][tau] = C_N(tau, xs[j], xs[0])
#         # with open(f"q1/case{case}/C_N_x{j+1}.npy", "wb") as f:
#         #     np.save(f, C_N_vals[j])
#         plt.figure()
#         plt.plot(C_N_vals[j].keys(), C_N_vals[j].values())
#         plt.xlabel(r"$\tau$")
#         plt.ylabel(r"$C_N(\tau,x_{},x_1)$".format(j+1))
#         plt.savefig(f"q1/case{case}_c0_2/plot_x{j+1}x1.png")


# plot_F_hat()
