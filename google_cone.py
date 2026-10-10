import math
import pyglet
from pyglet.gl import *
from pyglet.gl.gl_compat import GL_MODELVIEW, GL_PROJECTION, glBegin, glLoadIdentity, glMatrixMode, glRotatef, glTranslatef

# --- Window Setup ---
config = pyglet.gl.Config(double_buffer=True, depth_size=24)
window = pyglet.window.Window(width=1000, height=600, caption="Cone Unwrapping & Geodesic - Pyglet 3D", config=config)

# --- Geometry Parameters ---
L = 2.5          # Slant height of the cone
alpha = 0.5      # Base radius factor (r = L * alpha). 0.5 gives an exact half-circle (180 degrees)
h = math.sqrt(L**2 - (L*alpha)**2) # Height of the fully wrapped cone

# --- Geodesic Straight Line Parameters (in 2D Polar Space) ---
# A straight line in polar coordinates: R(theta) = d / cos(theta - theta_0)
d = 0.8
theta_0 = 0.5 * math.pi * 2 * alpha  # Centered symmetric line
theta_start = 0.05 * math.pi * 2 * alpha
theta_end = 0.95 * math.pi * 2 * alpha

# --- Animation State ---
time_elapsed = 0.0
flatten_factor = 0.0  # 0.0 = 3D Cone, 1.0 = Flat Half-Circle
dot_progress = 0.0    # 0.0 to 1.0 along the path

def flat_to_3d(R, theta, f):
    """
    Morphing math formula:
    Takes 2D Polar Coordinates (R, theta) on the flat sheet and returns (x, y, z)
    based on the current morph factor 'f' (0 = wrapped cone, 1 = flat sheet).
    """
    # Interpolate sector angle factor from alpha (0.5) to 1.0 (full half-circle)
    current_alpha = alpha + f * (1.0 - alpha)
    
    # Map the angle appropriately as the surface spreads open
    phi = theta * (alpha / current_alpha)
    
    sin_psi = current_alpha
    cos_psi = math.sqrt(max(0.0, 1.0 - sin_psi**2))
    
    x = R * sin_psi * math.cos(phi)
    y = R * sin_psi * math.sin(phi)
    z = R * cos_psi * (1.0 - f) - (h / 2.0 * (1.0 - f)) # Keep centered on Z-axis
    return x, y, z

def update(dt):
    global time_elapsed, flatten_factor, dot_progress
    time_elapsed += dt
    
    # Phase 1: Move the dot (0 to 4 seconds)
    if time_elapsed < 4.0:
        dot_progress = time_elapsed / 4.0
        flatten_factor = 0.0
    # Phase 2: Unwrap the cone (4 to 8 seconds)
    elif time_elapsed < 8.0:
        dot_progress = 1.0
        flatten_factor = (time_elapsed - 4.0) / 4.0
    # Phase 3: Hold the flat state (8 to 11 seconds) then reset loop
    elif time_elapsed < 11.0:
        dot_progress = 1.0
        flatten_factor = 1.0
    else:
        time_elapsed = 0.0

pyglet.clock.schedule_interval(update, 1/60.0)

@window.event
def on_draw():
    window.clear()
    
    # --- Setup 3D Projection Matrix ---
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window.width / float(window.height), 0.1, 100.0)  # pyright: ignore[reportUndefinedVariable]
    
    # --- Setup View Matrix ---
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Dynamically rotate camera into a top-down view when fully flattened (f=1.0)
    elev = 25.0 + flatten_factor * 65.0   # Morphs from 25 deg to 90 deg (Top down)
    azim = -60.0 + flatten_factor * -30.0 # Morphs from -60 deg to -90 deg
    
    # Distance to object
    glTranslatef(0, 0, -7.0)
    glRotatef(elev, 1, 0, 0)
    glRotatef(azim, 0, 0, 1)
    
    # Enable depth test and anti-aliasing smooth lines
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # --- Draw Cone Surface (Cyan Triangles) ---
    # Draw segments radiating outward to generate the surface mesh
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glColor4f(0.0, 0.8, 0.9, 0.4) # Transparent Cyan
    
    num_radial = 60
    num_slices = 20
    max_theta = 2.0 * math.pi * alpha
    
    for i in range(num_radial):
        t1 = (i / float(num_radial)) * max_theta
        t2 = ((i + 1) / float(num_radial)) * max_theta
        
        glBegin(GL_TRIANGLE_STRIP)
        for j in range(num_slices + 1):
            r_val = (j / float(num_slices)) * L
            
            x1, y1, z1 = flat_to_3d(r_val, t1, flatten_factor)
            x2, y2, z2 = flat_to_3d(r_val, t2, flatten_factor)
            
            glVertex3f(x1, y1, z1)
            glVertex3f(x2, y2, z2)
        glEnd()
        
    # --- Compute Geodesic Line Vertices ---
    line_points = []
    num_line_segments = 200
    for i in range(num_line_segments + 1):
        interp = i / float(num_line_segments)
        t = theta_start + interp * (theta_end - theta_start)
        r_val = d / math.cos(t - theta_0)
        if r_val <= L:
            line_points.append(flat_to_3d(r_val, t, flatten_factor))
            
    # --- Draw Geodesic Line (Red) ---
    glLineWidth(4.0)
    glColor3f(1.0, 0.1, 0.1) # Bright Red
    glBegin(GL_LINE_STRIP)
    for pt in line_points:
        glVertex3f(*pt)
    glEnd()
    
    # --- Draw Moving Dot (Black sphere/point) ---
    if line_points:
        dot_idx = int(dot_progress * (len(line_points) - 1))
        dx, dy, dz = line_points[dot_idx]
        
        # Draw a thick point acting as our moving dot
        glPointSize(12.0)
        glColor3f(0.0, 0.0, 0.0) # Black
        glBegin(GL_POINTS)
        glVertex3f(dx, dy, dz)
        glEnd()

# Start application loop
pyglet.app.run()
