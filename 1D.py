import numpy as np 
from scipy.special import erf 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

# Parameters from the lab
alpha = 1.0  # Diffusivity
ylim = 0.5   # Half-width of initial pulse (y_lim = 1/2)

# x range for plotting
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, 1000)  # High resolution for smooth plot

# Function for u(x, t)
def heat_solution(x, t, alpha, ylim):
    if t <= 0:
        # Initial condition: rectangular pulse
        return np.where(np.abs(x) <= ylim, 1.0, 0.0)
    else:
        sqrt_term = np.sqrt(4 * alpha * t)
        term1 = erf((ylim - x) / sqrt_term)
        term2 = erf((-ylim - x) / sqrt_term)
        return 0.5 * (term1 - term2)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.1, 1.1)  # Slightly below 0 for clarity
ax.set_xlabel('x')
ax.set_ylabel('Temperature u(x, t)')
ax.set_title('1D Heat Diffusion (α=1)')
ax.grid(True)

# Initial plot line
line, = ax.plot(x, heat_solution(x, 0, alpha, ylim), color='blue')

# Update function for animation
def update(frame):
    t = frame
    y = heat_solution(x, t, alpha, ylim)
    line.set_ydata(y)
    ax.set_title(f'1D Heat Diffusion (α=1) at t = {t:.3f}')
    return line,

# Time steps for animation (from t=0 to t=4, 200 frames)
t_max = 4.0
num_frames = 200
t_values = np.linspace(0, t_max, num_frames)

# Create animation
ani = FuncAnimation(fig, update, frames=t_values, interval=50, blit=True)

ani.save('heat_diffusion_1d.gif', writer='pillow', fps=60)
plt.show()