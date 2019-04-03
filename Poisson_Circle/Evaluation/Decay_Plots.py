import numpy as np
import matplotlib.pyplot as plt


def get_rate(step, init_rate=0.001, decay_step=10000, decay_rate=0.95, use_floor=False):
    if use_floor:
        decay_steps = np.floor(step/decay_step)
    else:
        decay_steps = step/decay_step
    return init_rate*np.power(decay_rate,decay_steps)

vget_rate = np.vectorize(get_rate)

config1 = [0.001, 10000, 0.95]
config2 = [0.00075, 10000, 0.95]
config3 = [0.001, 10000, 0.925]
config4 = [0.00075, 10000, 0.925]

configs = [config1, config2, config3, config4]

fig = plt.figure()
N = 1000000
M = 1000
steps = np.linspace(0,N,M)
use_log = False
for init_rate, decay_step, decay_rate in configs:
    vals = vget_rate(steps, init_rate=init_rate, decay_step=decay_step, decay_rate=decay_rate)
    plt.plot(steps, vals, label="IR: %.4f  DS: %d  DR: %.2f" %(init_rate, decay_step, decay_rate))
    if use_log:
        plt.yscale("log")
plt.legend()
plt.show()
    
