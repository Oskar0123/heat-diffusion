import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_min, x_max = 0, 7e6
y_min, y_max = 0, 7e6
x_step, y_step = 200, 200

#alpha = 1e-6

x = np.linspace(x_min, x_max, x_step)  # High resolution for smooth plot
y = np.linspace(y_min, y_max, y_step)
X, Y = np.meshgrid(x, y)

Y_MAX = 7e6
PLANET_CENTER = (0, 7e6)
COMET_RADIUS = 1e5
COMET_CENTER = [4.462e6, 4.462e6]
COMET_PEAK_TEMP = 10000.0

def get_dist(x_c, y_c):
    """Returns the distance from a specific center (x_c, y_c)."""
    return np.sqrt((X - x_c)**2 + (Y - (Y_MAX - y_c))**2)

radii = [6.360e6, 6.325e6, 6.300e6, 3.470e6, 1.210e6]
temps = [288.0, 1600.0, 3200.0, 5000.0, 6000.0]
diffs = [2e-6, 0.7e-6, 1.2e-6, 8.0e-6, 20.0e-6]
#temps = [288.0, 1600.0, 3200.0, 5000.0, 6000.0]
#diffs = [25e-6, 20e-6, 10e-6, 5e-6, 1e-6]
#diffs = [2e-6, 0.7e-6, 1.2e-6, 8.0e-6, 20.0e-6]
#0.8e-6, 0.7e-6, 1.2e-6, 8.0e-6, 20.0e-6

def cirk(R):
    return (X*X + (Y - y_max)**2) <= R**2

def comet(x_cord, y_cord, R):
    y_cord = 7 - y_cord
    return ((X - x_cord)**2 + (Y - y_cord)**2) <= R**2

fig, ax = plt.subplots(figsize=(16, 9)) # 8, 5
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)  # Slightly below 0 for clarity
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Earth Heat Diffusion')

#u_0_planet = np.zeros((x_step, y_step))
u_0_planet = np.full((x_step, y_step), 0.0)
alpha_map = np.full((x_step, y_step), 1e7) #1e-7
r_lim_map = np.zeros((x_step, y_step))

for r, t, a in zip(radii, temps, diffs):
    mask = get_dist(0, 0) <= r
    u_0_planet[mask] = t
    alpha_map[mask] = a
    r_lim_map[mask] = r
    
#alpha_map[get_dist(COMET_CENTER[0], COMET_CENTER[1]) <= 0.8e-6] = 0.1e-7 

def heat_solution(t):
    if t <= 0:
        # Initial condition: rectangular pulse  
        u = u_0_planet.copy()
        dist_c = get_dist(COMET_CENTER[0], COMET_CENTER[1])                
        u_comet_initial = COMET_PEAK_TEMP * np.exp(- (dist_c**2) / (2 * 0.2**2))
        u[dist_c <= 0.8e6] += u_comet_initial[dist_c <= 0.8e6]
        u[dist_c < COMET_RADIUS] = COMET_PEAK_TEMP
        
        return u    
    
    sqrt_term = np.sqrt(4 * alpha_map * t * 365 * 24 * 3600)
    #r = get_dist(0, 0)
    #r_lim = r_lim_map
    #u_planet = u_0_planet.copy()
    #u_planet = u_0_planet * 0.5 *  (erf((r_lim - r) / sqrt_term) - erf((-r_lim - r) / sqrt_term))
    
    r_c = get_dist(COMET_CENTER[0], COMET_CENTER[1])
    temp = COMET_PEAK_TEMP
    u_comet = temp * 0.5 *  (erf((COMET_RADIUS - r_c) / sqrt_term) - erf((-COMET_RADIUS - r_c) / sqrt_term))
    
    return u_0_planet + u_comet * np.exp(-(t*1e-10))

u = heat_solution(0)
surf = ax.imshow(u, extent=[x_min, x_max, y_min, y_max], cmap='jet', interpolation='nearest')
cbar = fig.colorbar(surf)
cbar.set_label('Temperature u(x, y, t)')

def update(frame):
    t = frame * 1e6
    u = heat_solution(t)
    surf.set_data(u)
    ax.set_title(f'2D Heat Diffusion at t = {frame:.3f}')
    return [surf]

delay_seconds = 5
fps = 30

hold_frame = int(delay_seconds * fps)

# Time steps for animation (from t=0 to t=4, 200 frames)
t_max = 1000
num_frames = t_max
t_values = np.linspace(0, t_max, num_frames)
t_delayed = np.concatenate([np.zeros(hold_frame), t_values])

# Create animation
ani = FuncAnimation(fig, update, frames=t_delayed, interval=50, blit=True)

ani.save('heat_diffusion_planet_test.gif', writer='pillow', fps=fps)
#ani.save('heat_diffusion_planet_super.gif', writer='pillow', fps=30)
#plt.show()