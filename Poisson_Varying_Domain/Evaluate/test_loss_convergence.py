import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import scipy.integrate
import scipy.optimize


CHECK_CAUCHY = False
COMPUTE_CAUCHY_MLE = False

if CHECK_CAUCHY:
    print(" Cauchy Tests |\n"+"-"*15+"\n")

    #gamma_start = 0.61199
    #gamma_stop  = 0.61201
    gamma_start = 0.61200308
    gamma_stop  = 0.61200309
    gamma_steps = 21
    for gamma in np.linspace(gamma_start, gamma_stop, gamma_steps):
        func = lambda x: np.power(gamma,2)/(np.power(x,2)+np.power(gamma,2)) * 1/np.sqrt(2*np.pi) * np.exp(-np.power(x,2)/2)
        info = scipy.integrate.quad(func, -np.inf, np.inf, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=5000)
        val = info[0]
        abserr = info[1]
        print("  {:.10f}\t{:7e}".format(gamma,val))

if COMPUTE_CAUCHY_MLE:
    
    integrand = lambda t: np.exp(-np.power(t,2))
    rhs = lambda g: np.sqrt(np.pi/2) * g * np.exp(np.power(g,2)/2)
    def func(g):
        info = scipy.integrate.quad(integrand, 0.0, g/np.sqrt(2.), args=(), full_output=0, epsabs=1.49e-10, epsrel=1.49e-10, limit=10000)
        integral = info[0]
        rhs_val = rhs(g)
        target = 0.5
        return np.power( target - rhs_val*(1 - 2/np.sqrt(np.pi)*integral) , 2)

    x0 = np.array([1.0])
    result = scipy.optimize.minimize(func, x0, method="BFGS", tol=1e-15)
    gamma_opt = result.x
    print("\nOptimized Gamma:")
    print(gamma_opt[0])


N = 75001

normal_loss = lambda x, s:  np.power(x,2)/(2*np.power(s,2)) + 0.5*np.log(2*np.pi*np.power(s,2))
laplace_loss = lambda x, b:  np.abs(x)/b + np.log(2*b)
cauchy_loss = lambda x, g: np.log( np.pi*g*(1.0 + np.power(x,2)/np.power(g,2)) )
quadratic_loss = lambda x: x*x

sigma = 1.0
samples = np.random.normal(0.0, sigma, [N])


normal_func = lambda theta: np.mean([normal_loss(x, theta) for x in samples])
laplace_func = lambda theta: np.mean([laplace_loss(x, theta) for x in samples])
cauchy_func = lambda theta: np.mean([cauchy_loss(x, theta) for x in samples])

uq_levels = np.array([0.1*n for n in range(1,30)])
uq_dict = dict()

for func, label in [[normal_func, "Normal"], [laplace_func, "Laplace"], [cauchy_func, "Cauchy"]]:
    theta_init = np.array([1.0])
    theta_opt, f_opt, d = fmin_l_bfgs_b(func, theta_init, bounds=[(0.0000001,100.0)], approx_grad=True, factr=1e1, pgtol=1e-10, maxls=60)

    print("\n" + label)
    print(theta_opt[0])
    #print(f_opt)
    #print(func(np.sqrt(2/np.pi)))

    array = np.zeros([uq_levels.size])
    for i, uq in enumerate(uq_levels):
        uq_success = np.sum(np.where(np.abs(samples) <= uq*theta_opt, 1, 0), axis=(0))
        array[i] = uq_success

    uq_dict[label] = 100.*array/samples.size

plt.figure()
for label in ["Normal", "Laplace", "Cauchy"]:
    plt.plot(uq_levels, uq_dict[label], label=label)
    
plt.legend(fontsize=24)
plt.show()    


#print(uq_dict["Cauchy"][-1])

"""
fig = plt.figure()
ax = fig.gca()
plt.plot(mesh, normal_vals, c='C0', label="Normal")
plt.plot(mesh, laplace_vals, c='C1', label="Laplace")
plt.plot(mesh, cauchy_vals, c='C2', label="Cauchy")
if PLOT_QUADRATIC:
    plt.plot(mesh, quadratic_vals, c='C3', label="Quadratic")
ax.legend(fontsize=24)
plt.show()
"""




