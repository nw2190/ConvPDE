import numpy as np
import matplotlib.pyplot as plt

N = 101
L = 2.0
mesh = np.linspace(-L,L,N)

# Plot quadratic loss for comparison
PLOT_QUADRATIC = True

# Plot sequence of hyperparameter values
PLOT_SEQ = True

normal_loss = lambda x, s:  np.power(x,2)/(2*np.power(s,2)) + 0.5*np.log(2*np.pi*np.power(s,2))
laplace_loss = lambda x, b:  np.abs(x)/b + np.log(2*b)
cauchy_loss = lambda x, g: np.log( np.pi*g*(1.0 + np.power(x,2)/np.power(g,2)) )
quadratic_loss = lambda x: x*x

theta = 0.5
normal_vals = np.array([normal_loss(x,theta) for x in mesh])
laplace_vals = np.array([laplace_loss(2*x,theta) for x in mesh])
cauchy_vals = np.array([cauchy_loss(2*x,theta) for x in mesh])
quadratic_vals = np.array([quadratic_loss(x) for x in mesh])


fig = plt.figure()
ax = fig.gca()
plt.plot(mesh, normal_vals, c='C0', label="Normal")
plt.plot(mesh, laplace_vals, c='C1', label="Laplace")
plt.plot(mesh, cauchy_vals, c='C2', label="Cauchy")
if PLOT_QUADRATIC:
    plt.plot(mesh, quadratic_vals, c='C3', label="Quadratic")
ax.legend(fontsize=24)
plt.show()

if PLOT_SEQ:
    
    normal_vals = np.array([normal_loss(x,theta) for x in mesh])
    laplace_vals = np.array([laplace_loss(2*x,theta) for x in mesh])
    cauchy_vals = np.array([cauchy_loss(2*x,theta) for x in mesh])

    for rate, title in [[0.9,"Decreasing"],[1.1,"Increasing"]]:
        for loss, label, color in [[normal_loss, "Normal", "C0"],[laplace_loss, "Laplace", "C1"],[cauchy_loss, "Cauchy", "C2"]]:
            if label == "Normal":
                theta = 1.0
            elif label == "Laplace":
                theta = 0.5
            else:
                theta = 0.5
            vals = np.array([loss(x,theta) for x in mesh])
            plt.plot(mesh, vals, c=color, label=label, linewidth=3.0)
            plt.plot(mesh, quadratic_vals, c='C3', label="Quadratic")

            for k in range(1,11):
                theta = theta*rate
                vals = np.array([loss(x,theta) for x in mesh])
                plt.plot(mesh, vals, c=color, linestyle="dashed", label=None)
            plt.title(title + " " + label + " Paramater", fontsize=24)
            plt.show()
