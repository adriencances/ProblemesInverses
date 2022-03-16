import numpy as np
from numpy.fft import ifftshift, fftshift, ifft, fft
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
import sys

# FIX SEED
np.random.seed(0)

N = 10000
if len(sys.argv) > 1:
    N = int(sys.argv[1])

c0 = 1
mu = np.array([0, -200, 0])
sigmas = np.array([100, 50, 100])
cov = np.diag(sigmas**2)


def F_hat(omega):
    return omega**2 * np.exp(-omega**2)

def F(t):
    return 1/(2*np.pi)*np.sqrt(np.pi)*(1/2-t**2/4)*np.exp(-t**2/4)


def test_F_hat_F():
    omegas = np.linspace(-3,3,100)
    d_omega = omegas[1] - omegas[0]
    ts = np.linspace(-10,10,100)
    integrand = F_hat(omegas)[:,None]*np.exp(-1j*omegas[:,None]*ts[None,:])
    vals = 1/(2*np.pi)*np.sum(integrand, axis=0)*d_omega
    F_vals = F(ts)
    plt.figure()
    plt.plot(ts, vals)
    plt.plot(ts, F_vals)
    plt.show()



def G_hat(omega, x, y):
    return 1 / (4*np.pi*norm(x-y)) * np.exp(1j * omega/c0*norm(x-y))

def plot_F_hat():
    omega = np.linspace(0,20,100)
    f_vals = F_hat(omega)
    plt.figure()
    plt.plot(omega, f_vals)
    plt.show()


omega = np.linspace(-10,10,201)
ys = np.random.multivariate_normal(mu ,cov, N)

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


def simulate_source():
    T = 10000
    tau_max = 100
    nb_samples = 1000*T//tau_max
    ts = np.linspace(0,T,nb_samples)
    ts -= T//2
    f_vals = F(ts)
    sqrt_fft_vals = np.sqrt(fft(fftshift(f_vals)))
    # Calcul par tranches de 100
    source = np.zeros((100, len(ts)))
    for id_y in tqdm(range(N)):
        Y = np.random.randn(len(ts))
        source[id_y%100] = np.real(ifft(sqrt_fft_vals * fft(Y)))
        if id_y%100 == 99:
            slice_id = id_y//100
            with open(f"u_vals/source/source_vals_slice_{slice_id}.npy", "wb") as f:
                np.save(f, source)

def simulate_u(T, nb_samples, new_N=None, cases=None):
    if new_N is None:
        new_N = N
    nb_slices = max(new_N // 100, 1)
    if cases is None:
        cases = [1,2,3]

    for case in cases:
        xs = get_xs(case)
        dist_to_xs = np.sum((ys[None,:,:] - xs[:,None,:])**2, axis=2)**0.5        # 5 x N
        u_vals_by_xs = [np.zeros((nb_samples)) for j in range(5)]
        slice_size = 100
        if new_N < N:
            slice_size = new_N
        for slice_id in tqdm(range(nb_slices)):
            source = np.load(open(f"u_vals/source/source_vals_slice_{slice_id}.npy", "rb"))
            if new_N < 100:
                source = source[:new_N]
            source = np.hstack([source, np.zeros((len(source),1))])
            if new_N < 100:
                dist_to_xs_slice = dist_to_xs[:,:new_N]
            else:
                dist_to_xs_slice = dist_to_xs[:,slice_id*100:(slice_id+1)*100]
            for j in range(5):
                factor = 1/(np.sqrt(new_N)*4*np.pi*dist_to_xs_slice[j])         # n
                ts = np.linspace(0, T, nb_samples)
                inds = ts[:,None] - dist_to_xs_slice[j][None,:]/c0          # nb_samples x n
                inds = np.floor(inds*nb_samples/T).astype(np.int64)
                inds[inds < 0] = -1
                idx_sources = np.repeat(np.arange(slice_size, dtype=np.int64)[None,:],
                                        nb_samples, axis=0)     # nb_samples x n
                source_vals = source[idx_sources.flatten(),inds.flatten()]
                source_vals = source_vals.reshape(inds.shape)   # nb_samples x n
                u_vals = np.sum(source_vals*factor[None,:], axis=1)
                u_vals_by_xs[j] += u_vals
        for j in range(5):
            with open(f"u_vals/case{case}/u_vals_N_{new_N}_x{j+1}.npy", "wb") as f:
                np.save(f, u_vals_by_xs[j])
            plt.figure()
            plt.plot(ts, u_vals_by_xs[j])
            plt.savefig(f"u_vals/case{case}/plot_u_vals_N_{new_N}_x{j+1}.png")
            plt.close()


def G_hat_vect(omega, x, y):
    _x = x.copy()
    _y = y.copy()
    if _y.ndim == 1:
        _y = _y[np.newaxis,:]
    dist_xy = norm(_x[None,:]-_y,axis=1)    # len(y)
    assert np.allclose(dist_xy, np.sqrt(np.sum((_x[None,:]-_y)**2, axis=1)))
    res =  1 / (4*np.pi*dist_xy[None,:]) * np.exp(1j * omega[:,None]/c0*dist_xy[None,:])
    return res      # len(omega) x len(y)


def K(y):
    _y = y.copy()
    if _y.ndim == 1:
        _y = _y[np.newaxis,:]
    exp_vals = np.exp(-(y - mu[np.newaxis,:])**2 / (2*sigmas**2)[np.newaxis,:])
    vals = exp_vals / (np.sqrt(2*np.pi)*sigmas)[np.newaxis,:]
    res = np.prod(vals, axis=1)
    return res


def plot_C_T_N(T, nb_samples):
    tau_max = 100
    ts = np.linspace(0, T, nb_samples)
    dt = ts[1] - ts[0]
    nb_taus = int(tau_max//dt)
    taus = np.array([k*dt for k in range(-nb_taus+1,nb_taus)])

    for case in [1,2,3]:
        u_vals = [None for j in range(5)]
        for j in range(5):
            vals = np.load(open(f"u_vals/case{case}/ESSAI_u_vals_x{j+1}.npy", "rb"))
            u_vals[j] = vals[:nb_samples].copy()

        C_T_N_vals = [None for j in range(5)]
        for j in tqdm(range(5)):
            vals_neg = np.array([np.mean(u_vals[j][k:] * u_vals[0][:-k or None])
                                 for k in range(nb_taus-1,0,-1)])*dt
            vals_pos = np.array([np.mean(u_vals[j][:-k or None] * u_vals[0][k:])
                                 for k in range(nb_taus)])*dt
            vals = np.concatenate([vals_neg, vals_pos], axis=0)
            C_T_N_vals[j] = vals
            with open(f"q3/case{case}/C_T_N_N{N}_T{int(T)}x{j+1}.npy", "wb") as f:
                np.save(f, C_T_N_vals[j])

        # val_min = min([C_T_N_vals[j].min() for j in range(5)])
        # val_max = max([C_T_N_vals[j].max() for j in range(5)])
        val_min = min([C_T_N_vals[j].min() for j in range(1)])
        val_max = max([C_T_N_vals[j].max() for j in range(1)])
        val_range = val_max - val_min
        for j in range(5):
            plt.figure()
            plt.plot(taus, C_T_N_vals[j])
            plt.ylim(val_min - 0.1*val_range, val_max + 0.1*val_range)
            plt.xlabel(r"$\tau$")
            plt.ylabel(rf"$C_{T,N}(\tau,x_{j+1},x_1)$")
            plt.savefig(f"q3/case{case}/plot_C_T_N_N{N}_T{int(T)}x{j+1}x1.png")
            plt.close()

# T = 10000
# simulate_source()

T = 10000
tau_max = 100
nb_samples = 1000*T//tau_max

# simulate_u(T, nb_samples)
# plot_C_T_N(T, nb_samples)
# for new_N in [100, 1000, 10000]:
for new_N in [1, 10, 50]:
    simulate_u(T, nb_samples, new_N)

# for T in [1000, 5000, 10000, 50000]:
# # for T in [5000]:
#     plot_C_T_N(T)

# for T in [100]:
#     plot_C_T_N(T)




# plt.figure()
# plt.plot(ts, process)
# plt.savefig("process_DFT.png")




