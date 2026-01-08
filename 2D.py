import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters from the lab
alpha = 2.0  # Diffusivity
#xlim = 0.5
#ylim = 0.5   # Half-width of initial pulse (y_lim = 1/2)

# x range for plotting
#x_min, x_max = 17.5 - 10, 17.5 + 10
#y_min, y_max = 37.5 - 10, 37.5 + 10
x_min, x_max = 0, 50
y_min, y_max = 0, 50
x_step, y_step = 500, 500
x = np.linspace(x_min, x_max, x_step)  # High resolution for smooth plot
y = np.linspace(y_min, y_max, y_step)
X, Y = np.meshgrid(x, y)

# Function for u(x, t)
def heat_solution(x, y, t, alpha):
    if t <= 0:
        u = np.zeros((x_step, y_step))
        initial_temp = 100.0
        u[(x - 17.5)**2 + (y - 37.5)**2 <= 4] = initial_temp
        return u        
    else:
        sqrt_term = np.sqrt(4 * alpha * t)
        r_0x = 17.5 # Center of initial pulse in x
        r_0y = 37.5 # Center of initial pulse in y
        delta_x = x
        delta_y = y
        r = np.sqrt((delta_x - r_0x)**2 + (delta_y - r_0y)**2)
        r_lim = 2.5 # Half-width of initial pulse in 2D
        alpha = 2.0  # Diffusivity
        term1 = erf((r_lim - r) / sqrt_term)
        term2 = erf((-r_lim - r) / sqrt_term)
        u_0 = 100.0  # Initial temperature
        return u_0 * 0.5 * (term1 - term2)
    
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)  # Slightly below 0 for clarity
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D Heat Diffusion (α=2)')
#ax.grid(True)

#result = heat_solution(X, Y, 0, alpha)
#print(result.shape)
#surf = ax.imshow(result, cmap='jet', interpolation='nearest')

# Initial plot line
u = heat_solution(X, Y, 0, alpha)
surf = ax.imshow(u, extent=[x_min, x_max, y_min, y_max], cmap='jet', interpolation='nearest')
cbar = fig.colorbar(surf)
cbar.set_label('Temperature u(x, y, t)')

# Update function for animation
def update(frame):
    t = frame
    u = heat_solution(X, Y, t, alpha)
    surf.set_data(u)
    ax.set_title(f'2D Heat Diffusion (α=2) at t = {t:.3f}')
    return [surf]

# Time steps for animation (from t=0 to t=4, 200 frames)
t_max = 10
num_frames = 60 * t_max
t_values = np.linspace(0, t_max, num_frames)

# Create animation
ani = FuncAnimation(fig, update, frames=t_values, interval=0.1, blit=True)

ani.save('heat_diffusion_2d.gif', writer='pillow', fps=60)
#plt.show()