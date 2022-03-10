from functools import total_ordering
import numpy as np
from numpy.fft import ifftshift, fftshift, ifft
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

# FIX SEED
np.random.seed(0)

N = 10000
c0 = 1
mu = np.array([0, -200, 0])
sigmas = np.array([100, 50, 100])
cov = np.diag(sigmas**2)


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
    plt.close()

    plt.figure()
    plt.scatter(ys[:,0], ys[:,1])
    plt.savefig("q1/plot_ys_dim12.png")
    plt.close()

    plt.figure()
    plt.scatter(ys[:,1], ys[:,2])
    plt.savefig("q1/plot_ys_dim23.png")
    plt.close()

    plt.figure()
    plt.scatter(ys[:,0], ys[:,2])
    plt.savefig("q1/plot_ys_dim13.png")
    plt.close()


omega = np.linspace(-10,10,201)
ys = np.random.multivariate_normal(mu ,cov, N)
# plot_ys(ys)
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
    d_omega = omega[1] - omega[0]
    omega_max = omega[-1]
    d_tau = 2*np.pi/(len(omega)*d_omega)
    tau_max = d_tau*len(omega)/2
    tau = np.linspace(-tau_max, tau_max, len(omega))
    for y in tqdm(ys):
        f_val = F_hat(omega)
        g_val1 = G_hat(omega, x1, y)
        g_val2 = G_hat(omega, x2, y)
        assert f_val.shape == g_val1.shape
        assert f_val.shape == g_val2.shape
        fourier_vals = f_val*np.conj(g_val1)*g_val2
        # fourier_vals *= np.exp(-1j*(omega+omega_max)*tau_max)
        int_vals = ifftshift(ifft(fftshift(fourier_vals)))
        # int_vals *= np.exp(1j*(tau_max*omega_max - tau*omega_max*d_tau))
        C_N_vals += np.real(int_vals)
    C_N_vals /= 2*np.pi*N
    return np.real(C_N_vals)


def gaussian_process(T, nb_samples):
    ts = np.linspace(0, T, nb_samples)
    X = np.random.randn(len(omega))
    Y = np.random.randn(len(omega))
    vals = X[:,None]*np.cos(omega[:,None]*ts[None,:])    # N*nb_samples
    vals += Y[:,None]*np.sin(omega[:,None]*ts[None,:])
    vals *= np.sqrt(F_hat(omega))[:,None]
    vals *= np.sqrt(omega[1]-omega[0])
    process = np.sum(vals, axis=0)
    return process

def simulate_several_processes():
    T = 10
    nb_samples = 100
    for essai in range(10):
        process = gaussian_process(T, nb_samples)
        ts = np.linspace(0,T,nb_samples)
        plt.figure()
        plt.plot(ts, process)
        plt.savefig(f"gaussian_process{essai}.png")

def simulate_source(T, nb_samples):
    source = []
    N = len(ys)
    for i in tqdm(range(N)):
        source.append(gaussian_process(T, nb_samples))
    return np.array(source)         # N x nb_samples

def simulate_u(xs, T, nb_samples):
    N = ys.shape[0]
    # source = np.random.rand(N, nb_samples)
    source = simulate_source(T, nb_samples)
    with open("source/source_vals.npy", "wb") as f:
        np.save(f, source)
    source = np.hstack([source, np.zeros((N,1))])
    u_vals_by_xs = []
    for j in tqdm(range(5)):
        x = xs[j]
        dist_to_x = np.sum((ys - x[None,:])**2, axis=1)**0.5
        factor = 1/(np.sqrt(N)*4*np.pi*dist_to_x)
        ts = np.linspace(0, T, nb_samples)
        inds = ts[:,None] - dist_to_x[None,:]/c0    # nb_samples x N
        inds = np.floor(inds*nb_samples/T)
        inds = inds.astype(np.int64)
        inds[inds < 0] = -1
        idx_sources = np.repeat(np.arange(N, dtype=np.int64)[None,:], nb_samples, axis=0)
        np.repeat(np.array([0,1,2])[:,None],4,axis=0)
        source_vals = source[idx_sources.flatten(),inds.flatten()]
        source_vals = source_vals.reshape(inds.shape)   # nb_samples x N
        u_vals = np.sum(source_vals*factor[None,:], axis=1)
        u_vals_by_xs.append(u_vals)
        with open(f"u_vals/u_vals_x{j+1}.npy", "wb") as f:
            np.save(f, u_vals)
        plt.figure()
        plt.plot(ts, u_vals)
        plt.savefig(f"u_vals_x{j+1}.png")
        plt.close()

    return u_vals_by_xs


def plot_ygrid(ygrid):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ygrid[:,0], ygrid[:,1], ygrid[:,2])
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    plt.savefig("plot_ygrid.png")

    plt.figure()
    plt.scatter(ygrid[:,0], ygrid[:,1])
    plt.savefig("q1/plot_ygrid_dim12.png")
    plt.close()

    plt.figure()
    plt.scatter(ygrid[:,1], ygrid[:,2])
    plt.savefig("q1/plot_ygrid_dim23.png")
    plt.close()

    plt.figure()
    plt.scatter(ygrid[:,0], ygrid[:,2])
    plt.savefig("q1/plot_ygrid_dim13.png")
    plt.close()


def G_hat_vect(omega, x, y):
    _x = x.copy()
    _y = y.copy()
    if _x.ndim == 1:
        _x = _x[np.newaxis,:]
    if _y.ndim == 1:
        _y = _y[np.newaxis,:]
    dist_xy = norm(_x-_y,axis=1)
    res =  1 / (4*np.pi*dist_xy[None,:]) * np.exp(1j * omega[:,None]/c0*dist_xy[None,:])
    return res      # len(omega) * len(y)


def K(y):
    _y = y.copy()
    if _y.ndim == 1:
        _y = _y[np.newaxis,:]
    exp_vals = np.exp(-(y - mu[np.newaxis,:])**2 / (2*sigmas**2)[np.newaxis,:])
    vals = exp_vals / (np.sqrt(2*np.pi)*sigmas)[np.newaxis,:]
    res = np.prod(vals, axis=1)
    return res


def C_1_ifft(x1, x2, taus, nb_pts_by_axis=100):
    interval1 = np.linspace(-300,300,nb_pts_by_axis)
    interval2 = np.linspace(-150,150,nb_pts_by_axis)
    interval3 = np.linspace(-300,300,nb_pts_by_axis)
    # volumes pour intégrer
    volume = (interval1[1]-interval1[0])*(interval2[1]-interval2[0])*(interval3[1]-interval3[0])
    d_omega = omega[1]-omega[0]

    dims = np.meshgrid(interval1, interval2, interval3, indexing="ij")
    ygrid = np.vstack([dim.flatten() for dim in dims]).transpose(1,0)       # nb_points x 3
    ygrid += mu[None,:]

    K_vals = K(ygrid)
    assert K_vals.min() >= 0
    assert np.abs(np.sum(K_vals)*volume - 1) < 0.01
    total_volume = (interval1[-1]-interval1[0])*(interval2[-1]-interval2[0])*(interval3[-1]-interval3[0])
    estimated_volume = volume*(nb_pts_by_axis-1)**3
    assert np.abs(total_volume - estimated_volume) < 0.01
    assert np.abs(estimated_volume - total_volume) < 0.01
    # return
    # plt.figure()
    # plt.scatter(ygrid[:,0], ygrid[:,1], c=K_vals)
    # plt.savefig("plot_K_ygrid.png")

    f_vals = F_hat(omega)
    g_vals1 =G_hat_vect(omega, x1, ygrid)
    g_vals2 = G_hat_vect(omega, x2, ygrid)

    # len(omega) x len(ygrid)
    integrand_vals = K_vals[np.newaxis,:]*f_vals[:,np.newaxis]*np.conj(g_vals1)*g_vals2
    imag_exp = np.exp(-1j*omega[:,np.newaxis]*taus[np.newaxis,:])   # len(omega) x len(taus)
    volume_int = np.sum(integrand_vals*volume,axis=1)           # len(omega)
    C_1_vals = np.sum(volume_int[:,np.newaxis]*d_omega*imag_exp, axis=0) / (2*np.pi)
    C_1_vals = np.real(C_1_vals)                                # len(taus)
    return C_1_vals



def plot_C_T_N():
    T = 200
    nb_samples = 500
    u_vals_by_xs = []
    for j in range(5):
        u_vals = np.load(open(f"u_vals/u_vals_x{j+1}.npy", "rb"))
        u_vals_by_xs.append(u_vals)
    ts = np.linspace(0, T, nb_samples)
    dt = ts[1] - ts[0]
    tau_max = 50
    C_T_N_vals_by_xs = []
    nb_taus = int(tau_max//dt)
    taus = [k*dt for k in range(-nb_taus+1,nb_taus)]
    for j in tqdm(range(5)):
        C_T_N_vals_neg = np.array([np.mean(u_vals_by_xs[j][k:] * u_vals_by_xs[0][:-k or None]) for k in range(nb_taus-1,0,-1)])
        C_T_N_vals_pos = np.array([np.mean(u_vals_by_xs[j][:-k or None] * u_vals_by_xs[0][k:]) for k in range(nb_taus)])
        C_T_N_vals = np.concatenate([C_T_N_vals_neg, C_T_N_vals_pos], axis=0)
        C_T_N_vals_by_xs.append(C_T_N_vals)
        plt.figure()
        plt.plot(taus, C_T_N_vals)
        plt.savefig(f"C_T_N_x{j+1}x1.png")

xs = get_xs(case=2)
taus = np.linspace(-20,20,100)
for j in tqdm(range(5)):
    C_1_vals = C_1_ifft(xs[j], xs[0], taus, nb_pts_by_axis=50)
    # plt.figure()
    # plt.plot(taus, C_1_vals)
    # plt.savefig(f"C_1_x{j+1}x1.png")



# xs = get_xs(case=2)
# T = 200
# nb_samples = 500
# # source = simulate_source(T, nb_samples)
# u_vals_by_xs = simulate_u(xs[0], T, nb_samples)

# plot_C_T_N()

# def f(t):
#     return np.exp(-t**2*100)


# f_vals = f(omega)

# f_vals = np.zeros(omega.shape)
# f_vals[60] = 1

# f_vals_ordered = fftshift(f_vals)
# f_vals_ordered_2 = np.roll(f_vals, -len(omega)//2)
# assert np.allclose(f_vals_ordered, f_vals_ordered_2)
# f_vals_ordered = f_vals
# g_vals = np.real(ifftshift(ifft(f_vals_ordered_2)))

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(f_vals)
# plt.subplot(3,1,2)
# plt.plot(f_vals_ordered_2)
# plt.subplot(3,1,3)
# plt.plot(g_vals)
# plt.show()



# xs = get_xs(case=2)
# for j in range(5):
#     C_N_vals = C_N_ifft(xs[j], xs[0])
#     plt.figure()
#     plt.plot(C_N_vals)
#     plt.savefig(f"Essai_C_N_x{j+1}x1.png")



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
