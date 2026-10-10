import math
import pyglet
from pyglet.math import Mat4, Vec3
from pyglet.gl import (
    Config, GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, 
    GL_TRIANGLES, GL_LINES, GL_POINTS, glEnable
)

# --- Modern Window Setup (Strict Core Profile) ---
config = Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3, forward_compatible=True)
window = pyglet.window.Window(width=1000, height=600, caption="Cone Unwrapping Modern Shader - Pyglet", config=config)

# --- Geometry Constants ---
L = 2.5          
alpha = 0.5      
h = math.sqrt(L**2 - (L*alpha)**2)

# Line parameters in 2D Polar Space
d = 0.8
theta_0 = 0.5 * math.pi * 2 * alpha
theta_start = 0.05 * math.pi * 2 * alpha
theta_end = 0.95 * math.pi * 2 * alpha

# --- GLSL Shaders ---
vertex_shader_source = """
#version 330 core
in vec2 r_theta; // Custom layout: x=Radius, y=Theta

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

uniform float f; // flatten_factor (0.0 = cone, 1.0 = flat half-circle)
uniform float h_val;
uniform float alpha_val;

void main() {
    float R = r_theta.x;
    float theta = r_theta.y;

    // GPU adaptation of the unrolling morph math
    float current_alpha = alpha_val + f * (1.0 - alpha_val);
    float phi = theta * (alpha_val / current_alpha);
    
    float sin_psi = current_alpha;
    float cos_psi = sqrt(max(0.0, 1.0 - sin_psi * sin_psi));
    
    float x = R * sin_psi * cos(phi);
    float y = R * sin_psi * sin(phi);
    float z = R * cos_psi * (1.0 - f) - (h_val / 2.0 * (1.0 - f));

    gl_Position = window.projection * window.view * vec4(x, y, z, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 fragColor;
uniform vec4 color;

void main() {
    fragColor = color;
}
"""

# Compile Shader Program
shader_program = pyglet.graphics.shader.ShaderProgram(
    pyglet.graphics.shader.Shader(vertex_shader_source, 'vertex'),
    pyglet.graphics.shader.Shader(fragment_shader_source, 'fragment')
)

# --- Build GPU VBO Buffers ---
batch = pyglet.graphics.Batch()

# 1. Cone Surface Geometry (Triangles)
cone_vertices = []
num_radial = 60
num_slices = 20
max_theta = 2.0 * math.pi * alpha

for i in range(num_radial):
    t1 = (i / float(num_radial)) * max_theta
    t2 = ((i + 1) / float(num_radial)) * max_theta
    for j in range(num_slices):
        r1 = (j / float(num_slices)) * L
        r2 = ((j + 1) / float(num_slices)) * L
        
        # Triangle 1
        cone_vertices.extend([r1, t1, r2, t1, r1, t2])
        # Triangle 2
        cone_vertices.extend([r2, t1, r2, t2, r1, t2])

cone_vertex_count = len(cone_vertices) // 2
cone_mesh = shader_program.vertex_list(
    cone_vertex_count, GL_TRIANGLES, batch=batch,
    r_theta=('f', cone_vertices)
)

# 2. Geodesic Line Geometry (Lines)
line_vertices = []
num_line_segments = 200
for i in range(num_line_segments + 1):
    interp = i / float(num_line_segments)
    t = theta_start + interp * (theta_end - theta_start)
    r_val = d / math.cos(t - theta_0)
    if r_val <= L:
        line_vertices.extend([r_val, t])

# Convert line nodes to continuous segment pairs for GL_LINES
line_pairs = []
for i in range(len(line_vertices) // 2 - 1):
    line_pairs.extend([line_vertices[2*i], line_vertices[2*i+1]])
    line_pairs.extend([line_vertices[2*i+2], line_vertices[2*i+3]])

line_vertex_count = len(line_pairs) // 2
line_mesh = shader_program.vertex_list(
    line_vertex_count, GL_LINES, batch=batch,
    r_theta=('f', line_pairs)
)

# --- Animation Control State ---
time_elapsed = 0.0
flatten_factor = 0.0
dot_progress = 0.0

def update(dt):
    global time_elapsed, flatten_factor, dot_progress
    time_elapsed += dt
    
    if time_elapsed < 4.0:
        dot_progress = time_elapsed / 4.0
        flatten_factor = 0.0
    elif time_elapsed < 8.0:
        dot_progress = 1.0
        flatten_factor = (time_elapsed - 4.0) / 4.0
    elif time_elapsed < 11.0:
        dot_progress = 1.0
        flatten_factor = 1.0
    else:
        time_elapsed = 0.0

pyglet.clock.schedule_interval(update, 1/60.0)

@window.event
def on_draw():
    window.clear()
    
    # Configure Modern Projection / View States Natively
    aspect = window.width / float(window.height)
    window.projection = Mat4.perspective_projection(aspect=aspect, z_near=0.1, z_far=100.0, fov=45.0)
    
    elev = 25.0 + flatten_factor * 65.0   
    azim = -60.0 + flatten_factor * -30.0 
    
    view_matrix = Mat4.from_translation(Vec3(0, 0, -7.0))
    view_matrix = view_matrix @ Mat4.from_rotation(math.radians(elev), Vec3(1, 0, 0))
    view_matrix = view_matrix @ Mat4.from_rotation(math.radians(azim), Vec3(0, 0, 1))
    window.view = view_matrix
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    pyglet.gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # FIXED: Uniforms updated via standard dict bracket overrides directly on the shader program context
    shader_program.use()
    shader_program['f'] = flatten_factor
    shader_program['h_val'] = h
    shader_program['alpha_val'] = alpha
    
    # Render Step 1: Draw Cyan Cone Surface
    shader_program['color'] = (0.0, 0.8, 0.9, 0.4)
    cone_mesh.draw(GL_TRIANGLES)
    
    # Render Step 2: Draw Red Geodesic Path
    shader_program['color'] = (1.0, 0.1, 0.1, 1.0)
    line_mesh.draw(GL_LINES)
    
    # Render Step 3: Draw Moving Point
    raw_nodes = len(line_vertices) // 2
    if raw_nodes > 0:
        dot_idx = min(int(dot_progress * (raw_nodes - 1)), raw_nodes - 1)
        dot_r = line_vertices[2 * dot_idx]
        dot_theta = line_vertices[2 * dot_idx + 1]
        
        dot_mesh = shader_program.vertex_list(1, GL_POINTS, r_theta=('f', [dot_r, dot_theta]))
        pyglet.gl.glPointSize(14.0)
        shader_program['color'] = (0.0, 0.0, 0.0, 1.0)
        dot_mesh.draw(GL_POINTS)

# Run modern application loop
pyglet.app.run()
