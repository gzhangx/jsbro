"""
A straight line on an unrolled cone wraps around the cone.

The 3D views are OpenGL. The cone is glass: the depth buffer keeps the
near line solid, and the wall tints the part of the line behind it.

Two windows open. The first shows the unrolled sectors beside the cone.
The second follows the dot. Drag either 3D view to turn it.

    python cone_wrap_demo.py
    python cone_wrap_demo.py -save
    python cone_wrap_demo.py -save movie.mp4

-save writes a movie, then exits: the dot walks the cone, walks back while
the cone unrolls, then walks the flat net in a straight line.
"""

import argparse
import os

import numpy as np
import pyglet
from pyglet.gl import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_BACK,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FRONT,
    GL_LEQUAL,
    GL_LESS,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PACK_ALIGNMENT,
    GL_RGB,
    GL_SCISSOR_TEST,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GLubyte,
    Config,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glEnable,
    glFinish,
    glPixelStorei,
    glReadPixels,
    glScissor,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

pyglet.options["shadow_window"] = False


# Three copies fill a half-plane. One straight line can cross at most that,
# so three is the most loops a single straight cut can show.
SECTOR_COPIES = 3
SLANT_LENGTH = 1.0
LINE_OFFSET = 0.62
# Opacity of each cone wall. 1 is solid, 0 is invisible.
CONE_ALPHA = 0.6
SAMPLES = 480
WALK_PER_SECOND = 36.0

PAPER = (0.965, 0.945, 0.906, 1.0)
INK = (0.141, 0.110, 0.078)
SECTOR_COLORS = (
    (0.965, 0.843, 0.659),
    (0.953, 0.894, 0.737),
    (0.906, 0.827, 0.631),
)
LAP_COLORS = (
    (0.769, 0.271, 0.180),
    (0.122, 0.478, 0.271),
    (0.141, 0.345, 0.651),
)
CONE_COLOR = (0.894, 0.765, 0.588)


def cone_angles(copies):
    """Sector angle and the cone's half-angle, for `copies` sectors in a half-plane."""
    beta = np.pi / copies
    # Arc of one sector equals the cone's base circumference: beta = 2 pi sin(alpha).
    alpha = np.arcsin(beta / (2.0 * np.pi))
    return alpha, beta


def straight_line_on_paper(length, offset, samples):
    """Horizontal chord at distance `offset` from the apex, clipped to the rim."""
    half = np.sqrt(length**2 - offset**2)
    abscissa = np.linspace(half, -half, samples)
    return abscissa, np.full_like(abscissa, offset)


def map_to_cone(x, y, alpha, beta):
    """Roll paper coordinates back onto the cone. The apex sits at the top."""
    slant = np.hypot(x, y)
    unrolled = np.arctan2(y, x)
    lap = np.floor(unrolled / beta).astype(int)
    local = unrolled - lap * beta
    phi = local * (2.0 * np.pi / beta)
    radius = slant * np.sin(alpha)
    height = SLANT_LENGTH * np.cos(alpha)
    cone_x = radius * np.cos(phi)
    cone_y = radius * np.sin(phi)
    cone_z = height - slant * np.cos(alpha)
    return cone_x, cone_y, cone_z, slant, unrolled, phi, lap


def _surface_normal(phi, alpha):
    return np.stack(
        (
            np.cos(alpha) * np.cos(phi),
            np.cos(alpha) * np.sin(phi),
            np.full_like(phi, np.sin(alpha)),
        ),
        axis=-1,
    )


def _cone_mesh(alpha, n_slant=48, n_phi=96):
    """Triangle mesh of the cone. Outward faces are counter-clockwise."""
    height = SLANT_LENGTH * np.cos(alpha)
    slant = np.linspace(0.0, SLANT_LENGTH, n_slant)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    positions = []
    normals = []
    colors = []
    cone = np.array(CONE_COLOR)

    def corner(s, p):
        radius = s * np.sin(alpha)
        return (
            np.array([radius * np.cos(p), radius * np.sin(p), height - s * np.cos(alpha)]),
            np.array([np.cos(alpha) * np.cos(p), np.cos(alpha) * np.sin(p), np.sin(alpha)]),
        )

    ds = SLANT_LENGTH / (n_slant - 1)
    dp = 2.0 * np.pi / n_phi
    for i, s in enumerate(slant[:-1]):
        for j, p in enumerate(phi):
            p2 = p + dp
            s2 = s + ds
            c00, n00 = corner(s, p)
            c01, n01 = corner(s, p2)
            c11, n11 = corner(s2, p2)
            c10, n10 = corner(s2, p)
            # Reversed so the outward side is the front face. The other
            # order made the wall toward the camera the one that was culled.
            for vertex, normal in (
                (c00, n00), (c11, n11), (c01, n01),
                (c00, n00), (c10, n10), (c11, n11),
            ):
                positions.extend(vertex.tolist())
                normals.extend(normal.tolist())
                colors.extend(cone.tolist())
    return positions, normals, colors


def _ribbon(points, normals, colors, half_width):
    """A strip lying on the surface, proud of it by the normals' own length."""
    count = len(points)
    left = np.empty((count, 3))
    right = np.empty((count, 3))
    for i in range(count):
        if i == 0:
            tangent = points[1] - points[0]
        elif i == count - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        side = np.cross(normals[i], tangent)
        side = side / (np.linalg.norm(side) + 1e-12)
        center = points[i] + normals[i]
        left[i] = center + side * half_width
        right[i] = center - side * half_width

    positions, out_normals, out_colors = [], [], []
    for i in range(count - 1):
        quad = (left[i], left[i + 1], right[i + 1], left[i], right[i + 1], right[i])
        normal_pair = (normals[i], normals[i + 1], normals[i + 1], normals[i], normals[i + 1], normals[i])
        color_pair = (colors[i], colors[i + 1], colors[i + 1], colors[i], colors[i + 1], colors[i])
        for vertex, normal, color in zip(quad, normal_pair, color_pair):
            positions.extend(vertex.tolist())
            out_normals.extend(normal.tolist())
            out_colors.extend(color.tolist())
    return positions, out_normals, out_colors


def _disk(radius, segments, z=0.0):
    positions, normals, colors = [], [], []
    normal = (0.0, 0.0, 1.0)
    color = (1.0, 1.0, 1.0)
    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        tri = (
            (0.0, 0.0, z),
            (radius * np.cos(a0), radius * np.sin(a0), z),
            (radius * np.cos(a1), radius * np.sin(a1), z),
        )
        for vertex in tri:
            positions.extend(vertex)
            normals.extend(normal)
            colors.extend(color)
    return positions, normals, colors


def _fan_mesh(beta, length, copies):
    positions, normals, colors = [], [], []
    steps = 24
    for lap in range(copies):
        color = SECTOR_COLORS[lap]
        a0 = np.pi - (lap + 1) * beta
        a1 = np.pi - lap * beta
        angles = np.linspace(a0, a1, steps)
        for i in range(steps - 1):
            tri = (
                (0.0, 0.0, 0.0),
                (length * np.cos(angles[i]), length * np.sin(angles[i]), 0.0),
                (length * np.cos(angles[i + 1]), length * np.sin(angles[i + 1]), 0.0),
            )
            for vertex in tri:
                positions.extend(vertex)
                normals.extend((0.0, 0.0, 1.0))
                colors.extend(color)
    return positions, normals, colors


def _paper_ribbon(x, y, lap, half_width):
    """Screen x is mirrored so the walk reads left to right."""
    points = np.stack((-x, y, np.zeros_like(x)), axis=1)
    count = len(points)
    positions, normals, colors = [], [], []
    for i in range(count - 1):
        color = LAP_COLORS[int(lap[i]) % len(LAP_COLORS)]
        y0 = points[i, 1]
        y1 = points[i + 1, 1]
        x0 = points[i, 0]
        x1 = points[i + 1, 0]
        quad = (
            (x0, y0 + half_width, 0.0),
            (x1, y1 + half_width, 0.0),
            (x1, y1 - half_width, 0.0),
            (x0, y0 + half_width, 0.0),
            (x1, y1 - half_width, 0.0),
            (x0, y0 - half_width, 0.0),
        )
        for vertex in quad:
            positions.extend(vertex)
            normals.extend((0.0, 0.0, 1.0))
            colors.extend(color)
    return positions, normals, colors


def _path_ribbon(cone, phi, lap, alpha, half_width, lift):
    points = np.stack(cone, axis=1)
    normals = _surface_normal(phi, alpha) * lift
    colors = np.array([LAP_COLORS[int(k) % len(LAP_COLORS)] for k in lap])
    return _ribbon(points, normals, colors, half_width)


VERTEX_SRC = """#version 330 core
in vec3 position;
in vec3 normal;
in vec3 color;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_normal = mat3(u_model) * normal;
    v_color = color;
    gl_Position = u_mvp * world;
}
"""

FRAGMENT_SRC = """#version 330 core
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;

uniform vec3 u_light;
uniform vec3 u_camera;
uniform float u_lit;
uniform float u_alpha;

out vec4 frag_color;

void main() {
    vec3 color = v_color;
    if (u_lit > 0.5) {
        vec3 n = normalize(v_normal);
        if (!gl_FrontFacing) n = -n;
        vec3 light = normalize(u_light);
        float diffuse = max(dot(n, light), 0.0);
        vec3 view = normalize(u_camera - v_world);
        vec3 half_dir = normalize(light + view);
        float spec = pow(max(dot(n, half_dir), 0.0), 28.0);
        color = color * (0.5 + 0.62 * diffuse) + vec3(spec * 0.14);
    }
    frag_color = vec4(color, u_alpha);
}
"""


def _upload(program, positions, normals, colors):
    count = len(positions) // 3
    return program.vertex_list(
        count,
        GL_TRIANGLES,
        position=("f", positions),
        normal=("f", normals),
        color=("f", colors),
    )


class Renderer:
    """GPU meshes for one OpenGL context."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.program = None
        self.meshes = {}

    def build(self):
        if self.program is not None:
            return
        self.program = ShaderProgram(
            Shader(VERTEX_SRC, "vertex"),
            Shader(FRAGMENT_SRC, "fragment"),
        )
        g = self.geometry
        self.meshes = {
            "cone": _upload(self.program, *g["cone"]),
            "path": _upload(self.program, *g["path"]),
            "seam": _upload(self.program, *g["seam"]),
            "fan": _upload(self.program, *g["fan"]),
            "paper": _upload(self.program, *g["paper"]),
            "dot": _upload(self.program, *_sphere(0.012, 10, 14)),
            "paper_dot": _upload(self.program, *_disk(0.045, 20, 0.02)),
        }
        self.program["u_light"] = (0.35, -0.55, 0.76)
        self.program["u_model"] = Mat4()
        self.program["u_alpha"] = 1.0

    def draw(self, name, mvp, camera, lit, model=None, alpha=1.0):
        self.program["u_mvp"] = mvp
        self.program["u_model"] = Mat4() if model is None else model
        self.program["u_camera"] = camera
        self.program["u_lit"] = float(lit)
        self.program["u_alpha"] = float(alpha)
        self.program.use()
        self.meshes[name].draw(GL_TRIANGLES)


def _sphere(radius, stacks, slices):
    positions, normals, colors = [], [], []
    color = (1.0, 1.0, 1.0)
    for i in range(stacks):
        v0 = np.pi * i / stacks
        v1 = np.pi * (i + 1) / stacks
        for j in range(slices):
            u0 = 2.0 * np.pi * j / slices
            u1 = 2.0 * np.pi * (j + 1) / slices

            def vertex(u, v):
                n = np.array([
                    np.sin(v) * np.cos(u),
                    np.sin(v) * np.sin(u),
                    np.cos(v),
                ])
                return n * radius, n

            a, na = vertex(u0, v0)
            b, nb = vertex(u1, v0)
            c, nc = vertex(u1, v1)
            d, nd = vertex(u0, v1)
            for point, normal in ((a, na), (b, nb), (c, nc), (a, na), (c, nc), (d, nd)):
                positions.extend(point.tolist())
                normals.extend(normal.tolist())
                colors.extend(color)
    return positions, normals, colors


def _translate(x, y, z):
    return Mat4.from_translation(Vec3(float(x), float(y), float(z)))


def _look(eye, target, up=(0.0, 0.0, 1.0)):
    return Mat4.look_at(Vec3(*eye), Vec3(*target), Vec3(*up))


def _rotate(vector, axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


class Demo:
    def __init__(self):
        alpha, beta = cone_angles(SECTOR_COPIES)
        paper_x, paper_y = straight_line_on_paper(SLANT_LENGTH, LINE_OFFSET, SAMPLES)
        cone_x, cone_y, cone_z, slant, unrolled, phi, lap = map_to_cone(
            paper_x, paper_y, alpha, beta
        )
        height = SLANT_LENGTH * np.cos(alpha)
        path = _path_ribbon((cone_x, cone_y, cone_z), phi, lap, alpha, 0.0075, 0.0045)
        seam_phi = np.linspace(0.0, 0.0, 2)
        seam_s = np.array([0.0, SLANT_LENGTH])
        seam_pts = np.stack(
            (
                seam_s * np.sin(alpha),
                np.zeros(2),
                height - seam_s * np.cos(alpha),
            ),
            axis=1,
        )
        seam_n = np.repeat(_surface_normal(np.array([0.0]), alpha), 2, axis=0) * 0.0025
        seam = _ribbon(
            seam_pts,
            seam_n,
            np.tile(INK, (2, 1)),
            0.0035,
        )
        geometry = {
            "cone": _cone_mesh(alpha),
            "path": path,
            "seam": seam,
            "fan": _fan_mesh(beta, SLANT_LENGTH, SECTOR_COPIES),
            "paper": _paper_ribbon(paper_x, paper_y, lap, 0.018),
        }
        self.alpha = alpha
        self.height = height
        self.paper = np.stack((-paper_x, paper_y, np.zeros_like(paper_x)), axis=1)
        self.cone = np.stack((cone_x, cone_y, cone_z), axis=1)
        self.phi = phi
        self.slant = slant
        self.unrolled = unrolled
        self.lap = lap
        self.beta = beta
        self.cursor = 0.0
        self.paused = False
        self.yaw = np.deg2rad(-62.0)
        self.pitch = np.deg2rad(24.0)
        self.distance = 2.35
        self.follow_orbit = None
        self.dragging = None

        config = Config(depth_size=24, double_buffer=True, major_version=3, minor_version=3)
        self.main = pyglet.window.Window(
            1360, 760, caption="Unrolled cone", resizable=True, config=config,
        )
        self.follow = pyglet.window.Window(
            760, 760, caption="Following the dot", resizable=True, config=config,
        )
        ink = tuple(int(c * 255) for c in INK) + (255,)
        muted = (92, 81, 70, 255)
        self.main.switch_to()
        self.gpu_main = Renderer(geometry)
        self.gpu_main.build()
        self.main_labels = {
            "title": pyglet.text.Label(
                "A straight line on the unrolled cone wraps around the cone",
                font_size=15, color=ink, anchor_x="center", anchor_y="top",
            ),
            "paper": pyglet.text.Label(
                "Unrolled  ·  the path is one straight line",
                font_size=12, color=ink, anchor_x="center", anchor_y="top",
            ),
            "cone": pyglet.text.Label(
                "Rolled back up  ·  the cone is glass",
                font_size=12, color=ink, anchor_x="center", anchor_y="top",
            ),
            "hint": pyglet.text.Label(
                "Drag the cone to turn it.  The glass lets the far side of the line show through.",
                font_size=11, color=muted, anchor_x="center", anchor_y="bottom",
            ),
            "status": pyglet.text.Label(
                "", font_size=12, color=ink, anchor_x="center", anchor_y="bottom",
            ),
        }
        self.follow.switch_to()
        self.gpu_follow = Renderer(geometry)
        self.gpu_follow.build()
        self.follow_labels = {
            "title": pyglet.text.Label(
                "", font_size=14, color=ink, anchor_x="center", anchor_y="top",
            ),
            "status": pyglet.text.Label(
                "", font_size=12, color=ink, anchor_x="center", anchor_y="bottom",
            ),
        }
        self._bind()

    def _bind(self):
        demo = self

        @self.main.event
        def on_draw():
            demo.draw_main()

        @self.follow.event
        def on_draw():
            demo.draw_follow()

        @self.main.event
        def on_mouse_press(x, y, button, modifiers):
            if x >= demo.main.width * 0.5:
                demo.dragging = "main"

        @self.main.event
        def on_mouse_release(x, y, button, modifiers):
            demo.dragging = None

        @self.main.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if demo.dragging == "main":
                demo.yaw += dx * 0.008
                demo.pitch = float(np.clip(demo.pitch + dy * 0.006, -1.2, 1.2))

        @self.main.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            if x >= demo.main.width * 0.5:
                demo.distance = float(np.clip(demo.distance - scroll_y * 0.12, 1.2, 5.0))

        @self.follow.event
        def on_mouse_press(x, y, button, modifiers):
            demo.dragging = "follow"
            if demo.follow_orbit is None:
                demo.follow_orbit = [0.0, 0.0]

        @self.follow.event
        def on_mouse_release(x, y, button, modifiers):
            demo.dragging = None
            demo.follow_orbit = None

        @self.follow.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if demo.follow_orbit is not None:
                demo.follow_orbit[0] += dx * 0.008
                demo.follow_orbit[1] = float(np.clip(
                    demo.follow_orbit[1] + dy * 0.006, -0.8, 0.8
                ))

        @self.main.event
        def on_close():
            pyglet.app.exit()

        @self.follow.event
        def on_close():
            pyglet.app.exit()

        @self.main.event
        def on_key_press(symbol, modifiers):
            demo._key(symbol)

        @self.follow.event
        def on_key_press(symbol, modifiers):
            demo._key(symbol)

    def _key(self, symbol):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused

    def advance(self, dt):
        if not self.paused:
            self.cursor = (self.cursor + dt * WALK_PER_SECOND) % SAMPLES

    @property
    def frame(self):
        return int(self.cursor) % SAMPLES

    def _status(self):
        i = self.frame
        wound = np.rad2deg((self.unrolled[i] - self.unrolled[0]) * (2.0 * np.pi / self.beta))
        return (
            f"distance from apex  {self.slant[i]:.2f}     "
            f"loop {int(self.lap[i]) + 1} of {SECTOR_COPIES}     "
            f"wound {wound:.0f}° around the cone"
        )

    def _overview_eye(self):
        target = np.array([0.0, 0.0, self.height * 0.42])
        eye = target + self.distance * np.array([
            np.cos(self.pitch) * np.cos(self.yaw),
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
        ])
        return eye, target

    def _follow_eye(self):
        """Sit just behind the dot and look along the direction it is walking."""
        i = self.frame
        pos = self.cone[i]
        ahead = min(i + 4, len(self.cone) - 1)
        behind = max(i - 4, 0)
        forward = self.cone[ahead] - self.cone[behind]
        length = np.linalg.norm(forward)
        if length < 1e-8:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / length
        normal = _surface_normal(np.array([float(self.phi[i])]), self.alpha)[0]
        if self.follow_orbit is not None:
            yaw, pitch = self.follow_orbit
            forward = _rotate(forward, normal, yaw)
            normal = _rotate(normal, forward, -pitch)
        eye = pos - forward * 0.92 + normal * 0.18
        target = pos + forward * 0.55
        return eye, target, normal

    def _paint_3d(self, gpu, eye, target, aspect, up=(0.0, 0.0, 1.0)):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(True)
        glDisable(GL_BLEND)
        proj = Mat4.perspective_projection(aspect, 0.02, 40.0, fov=34.0)
        view = _look(eye, target, up)
        mvp = proj @ view
        camera = (float(eye[0]), float(eye[1]), float(eye[2]))
        # The line is drawn first and writes depth. The glass is drawn
        # afterward without writing depth, so it tints the far line and
        # leaves the nearer line solid.
        glDisable(GL_CULL_FACE)
        gpu.draw("seam", mvp, camera, lit=0.0)
        gpu.draw("path", mvp, camera, lit=0.0)
        i = self.frame
        pos = self.cone[i]
        normal = _surface_normal(np.array([self.phi[i]]), self.alpha)[0]
        dot = pos + normal * 0.02
        color = LAP_COLORS[int(self.lap[i]) % len(LAP_COLORS)]
        gpu.meshes["dot"].color[:] = np.tile(color, len(gpu.meshes["dot"].color) // 3)
        gpu.draw("dot", mvp, camera, lit=1.0, model=_translate(*dot))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        gpu.draw("cone", mvp, camera, lit=1.0, alpha=CONE_ALPHA)
        glCullFace(GL_BACK)
        gpu.draw("cone", mvp, camera, lit=1.0, alpha=CONE_ALPHA)
        glDepthMask(True)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)

    def _begin(self, window, gpu):
        window.switch_to()
        gpu.build()
        glClearColor(*PAPER)
        fb_w, fb_h = window.get_framebuffer_size()
        glViewport(0, 0, fb_w, fb_h)
        glDisable(GL_SCISSOR_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        return fb_w, fb_h

    def draw_main(self):
        fb_w, fb_h = self._begin(self.main, self.gpu_main)
        split = int(fb_w * 0.50)
        glEnable(GL_SCISSOR_TEST)

        glViewport(0, 0, max(split, 1), fb_h)
        glScissor(0, 0, max(split, 1), fb_h)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        ortho = Mat4.orthogonal_projection(-1.35, 1.35, -0.42, 1.28, -1.0, 1.0)
        camera = (0.0, 0.0, 1.0)
        self.gpu_main.draw("fan", ortho, camera, lit=0.0)
        self.gpu_main.draw("paper", ortho, camera, lit=0.0)
        i = self.frame
        color = LAP_COLORS[int(self.lap[i]) % len(LAP_COLORS)]
        self.gpu_main.meshes["paper_dot"].color[:] = np.tile(
            color, len(self.gpu_main.meshes["paper_dot"].color) // 3
        )
        self.gpu_main.draw(
            "paper_dot", ortho, camera, lit=0.0, model=_translate(*self.paper[i]),
        )

        glViewport(split, 0, fb_w - split, fb_h)
        glScissor(split, 0, fb_w - split, fb_h)
        glClear(GL_DEPTH_BUFFER_BIT)
        eye, target = self._overview_eye()
        aspect = max(fb_w - split, 1) / max(fb_h, 1)
        self._paint_3d(self.gpu_main, eye, target, aspect)

        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, fb_w, fb_h)
        w, h = self.main.get_size()
        labels = self.main_labels
        labels["title"].x = w * 0.5
        labels["title"].y = h - 12
        labels["paper"].x = w * 0.25
        labels["paper"].y = h - 46
        labels["cone"].x = w * 0.75
        labels["cone"].y = h - 46
        labels["status"].x = w * 0.5
        labels["status"].y = 36
        labels["status"].text = self._status()
        labels["hint"].x = w * 0.5
        labels["hint"].y = 12
        for name in ("title", "paper", "cone", "status", "hint"):
            labels[name].draw()

    def draw_follow(self):
        fb_w, fb_h = self._begin(self.follow, self.gpu_follow)
        eye, target, up = self._follow_eye()
        self._paint_3d(
            self.gpu_follow, eye, target, max(fb_w, 1) / max(fb_h, 1), up=up,
        )
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, fb_w, fb_h)
        w, h = self.follow.get_size()
        i = self.frame
        labels = self.follow_labels
        labels["title"].x = w * 0.5
        labels["title"].y = h - 16
        labels["title"].text = (
            f"Following the dot  ·  loop {int(self.lap[i]) + 1} of {SECTOR_COPIES}"
        )
        labels["status"].x = w * 0.5
        labels["status"].y = 16
        labels["status"].text = self._status()
        labels["title"].draw()
        labels["status"].draw()

    def run(self):
        pyglet.clock.schedule_interval(lambda dt: self.advance(dt), 1.0 / 60.0)
        pyglet.app.run()


def _smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _morph_points(slant, phi, lap, t, alpha, beta, height):
    """Unroll one wound sheet. `lap` counts extra turns, so the angle never jumps.

    The cut at angle 0 stays put. The rest of the sheet swings open around it
    until the paper is flat and the chord is a straight line.
    """
    slant = np.asarray(slant, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    lap = np.asarray(lap, dtype=np.float64)
    turn = phi + lap * (2.0 * np.pi)
    flat_angle = turn * (beta / (2.0 * np.pi))
    angle = (1.0 - t) * turn + t * flat_angle
    radius = ((1.0 - t) * np.sin(alpha) + t) * slant
    z = (1.0 - t) * (height - slant * np.cos(alpha))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)


def _morph_normals(phi, lap, t, alpha, beta):
    turn = np.asarray(phi, dtype=np.float64) + np.asarray(lap, dtype=np.float64) * (2.0 * np.pi)
    angle = turn * ((1.0 - t) + t * (beta / (2.0 * np.pi)))
    cone = _surface_normal(angle, alpha)
    normals = (1.0 - t) * cone
    normals = np.array(normals, copy=True)
    normals[..., 2] += t
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    return normals


def _wound_sheet(n_slant, n_around, turns):
    """One mesh wound `turns` times. The first and last edges are the single cut."""
    slant_rows = np.linspace(0.0, SLANT_LENGTH, n_slant)
    n_phi = n_around * turns
    psi_cols = np.linspace(0.0, turns * 2.0 * np.pi, n_phi, endpoint=False)
    ds = SLANT_LENGTH / (n_slant - 1)
    dp = (turns * 2.0 * np.pi) / n_phi
    s, p = np.meshgrid(slant_rows[:-1], psi_cols, indexing="ij")
    s = s.ravel()
    p = p.ravel()
    # Same winding as `_cone_mesh`: outward side is the front face.
    s_vert = np.stack((s, s + ds, s, s, s + ds, s + ds), axis=1).ravel()
    p_vert = np.stack((p, p + dp, p + dp, p, p, p + dp), axis=1).ravel()
    return s_vert, p_vert


def _sheet_colors(count, t):
    cone = np.array(CONE_COLOR, dtype=np.float64)
    paper = np.array(SECTOR_COLORS[1], dtype=np.float64)
    color = (1.0 - t) * cone + t * paper
    return np.tile(color, (count, 1))


def _assign_mesh(mesh, positions, normals, colors):
    mesh.position[:] = np.ascontiguousarray(positions, dtype=np.float32).ravel()
    mesh.normal[:] = np.ascontiguousarray(normals, dtype=np.float32).ravel()
    mesh.color[:] = np.ascontiguousarray(colors, dtype=np.float32).ravel()


def _empty_mesh(program, count):
    zeros = [0.0] * (count * 3)
    return program.vertex_list(
        count,
        GL_TRIANGLES,
        position=("f", zeros),
        normal=("f", zeros),
        color=("f", zeros),
    )


class _UnwrapFilm:
    """One 3D view that rolls the cone out into the straight-line net."""

    def __init__(self):
        alpha, beta = cone_angles(SECTOR_COPIES)
        paper_x, paper_y = straight_line_on_paper(SLANT_LENGTH, LINE_OFFSET, SAMPLES)
        _cx, _cy, _cz, slant, _unrolled, phi, lap = map_to_cone(
            paper_x, paper_y, alpha, beta
        )
        self.alpha = alpha
        self.beta = beta
        self.height = SLANT_LENGTH * np.cos(alpha)
        self.path_s = slant
        self.path_phi = phi
        self.path_lap = lap
        self.path_color = np.array(
            [LAP_COLORS[int(k) % len(LAP_COLORS)] for k in lap], dtype=np.float64
        )
        self.sheet_s, self.sheet_phi = _wound_sheet(36, 64, SECTOR_COPIES)
        self.sheet_lap = np.zeros_like(self.sheet_phi)
        self.seam_s = np.linspace(0.0, SLANT_LENGTH, 24)
        self.window = None
        self.program = None
        self.sheet = None
        self.path = None
        self.seam = None
        self.dot = None
        self.caption = None

    def open(self, width=1280, height=720):
        config = Config(depth_size=24, double_buffer=True, major_version=3, minor_version=3)
        self.window = pyglet.window.Window(
            width, height, caption="Saving cone unwrap", resizable=False, vsync=False, config=config,
        )
        self.window.switch_to()

        @self.window.event
        def on_draw():
            pass

        self.program = ShaderProgram(Shader(VERTEX_SRC, "vertex"), Shader(FRAGMENT_SRC, "fragment"))
        self.program["u_light"] = (0.35, -0.55, 0.76)
        self.program["u_model"] = Mat4()
        self.program["u_alpha"] = 1.0
        self.sheet = _empty_mesh(self.program, len(self.sheet_s))
        path_verts = self._path_arrays(0.0)[0].shape[0]
        self.path = _empty_mesh(self.program, path_verts)
        seam_verts = self._seam_arrays(0.0)[0].shape[0]
        self.seam = _empty_mesh(self.program, seam_verts)
        dot_positions, dot_normals, dot_colors = _sphere(0.02, 12, 16)
        self.dot = _upload(self.program, dot_positions, dot_normals, dot_colors)
        ink = tuple(int(c * 255) for c in INK) + (255,)
        self.caption = pyglet.text.Label(
            "", font_size=16, color=ink, anchor_x="center", anchor_y="top",
        )
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None

    def _pose(self, blend):
        """Cone overview, then a high view while the sheet sweeps open, then the flat net."""
        yaw = np.deg2rad(-62.0)
        pitch = np.deg2rad(24.0)
        distance = 2.45
        target0 = np.array([0.0, 0.0, self.height * 0.42])
        eye0 = target0 + distance * np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])
        # The single sheet swings through more than a full turn, so this
        # view stays high enough to see the whole disk until it is flat.
        eye_mid = np.array([0.0, -0.20, 4.60])
        target_mid = np.array([0.0, 0.0, 0.0])
        eye1 = np.array([0.0, -1.15, 2.45])
        target1 = np.array([0.0, 0.40, 0.0])
        split = 0.92
        if blend <= split:
            b = _smoothstep(blend / split) if blend > 0.0 else 0.0
            eye = (1.0 - b) * eye0 + b * eye_mid
            target = (1.0 - b) * target0 + b * target_mid
        else:
            b = _smoothstep((blend - split) / (1.0 - split))
            eye = (1.0 - b) * eye_mid + b * eye1
            target = (1.0 - b) * target_mid + b * target1
        return eye, target

    def _path_arrays(self, t):
        points = _morph_points(
            self.path_s, self.path_phi, self.path_lap, t, self.alpha, self.beta, self.height,
        )
        normals = _morph_normals(
            self.path_phi, self.path_lap, t, self.alpha, self.beta,
        ) * 0.006
        positions, out_normals, out_colors = _ribbon(points, normals, self.path_color, 0.009)
        return (
            np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            np.asarray(out_normals, dtype=np.float32).reshape(-1, 3),
            np.asarray(out_colors, dtype=np.float32).reshape(-1, 3),
        )

    def _seam_arrays(self, t):
        positions, normals, colors = [], [], []
        for edge in (0.0, SECTOR_COPIES * 2.0 * np.pi):
            phi = np.full_like(self.seam_s, edge)
            laps = np.zeros_like(self.seam_s)
            points = _morph_points(
                self.seam_s, phi, laps, t, self.alpha, self.beta, self.height,
            )
            lifted = _morph_normals(phi, laps, t, self.alpha, self.beta) * 0.002
            ink = np.tile(np.array(INK, dtype=np.float64), (len(self.seam_s), 1))
            piece = _ribbon(points, lifted, ink, 0.004)
            positions.extend(piece[0])
            normals.extend(piece[1])
            colors.extend(piece[2])
        return (
            np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            np.asarray(normals, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.float32).reshape(-1, 3),
        )

    def _upload_frame(self, t):
        positions = _morph_points(
            self.sheet_s, self.sheet_phi, self.sheet_lap, t, self.alpha, self.beta, self.height,
        )
        normals = _morph_normals(self.sheet_phi, self.sheet_lap, t, self.alpha, self.beta)
        colors = _sheet_colors(len(self.sheet_s), t)
        _assign_mesh(self.sheet, positions, normals, colors)
        _assign_mesh(self.path, *self._path_arrays(t))
        _assign_mesh(self.seam, *self._seam_arrays(t))

    def _draw_mesh(self, mesh, mvp, camera, lit, model=None, alpha=1.0):
        self.program["u_mvp"] = mvp
        self.program["u_model"] = Mat4() if model is None else model
        self.program["u_camera"] = (float(camera[0]), float(camera[1]), float(camera[2]))
        self.program["u_lit"] = float(lit)
        self.program["u_alpha"] = float(alpha)
        self.program.use()
        mesh.draw(GL_TRIANGLES)

    def draw(self, t, index, blend, caption):
        self.window.switch_to()
        self._upload_frame(t)
        fb_w, fb_h = self.window.get_framebuffer_size()
        glViewport(0, 0, fb_w, fb_h)
        glDisable(GL_SCISSOR_TEST)
        glClearColor(*PAPER)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        eye, target = self._pose(blend)
        aspect = max(fb_w, 1) / max(fb_h, 1)
        proj = Mat4.perspective_projection(aspect, 0.02, 40.0, fov=32.0)
        view = _look(eye, target, (0.0, 0.0, 1.0))
        mvp = proj @ view

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(True)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        self._draw_mesh(self.seam, mvp, eye, lit=0.0)
        self._draw_mesh(self.path, mvp, eye, lit=0.0)
        point = _morph_points(
            self.path_s[index], self.path_phi[index], self.path_lap[index],
            t, self.alpha, self.beta, self.height,
        )
        normal = _morph_normals(
            np.array([self.path_phi[index]]), np.array([self.path_lap[index]]),
            t, self.alpha, self.beta,
        )[0]
        dot = point + normal * 0.025
        color = LAP_COLORS[int(self.path_lap[index]) % len(LAP_COLORS)]
        self.dot.color[:] = np.tile(color, len(self.dot.color) // 3)
        self._draw_mesh(self.dot, mvp, eye, lit=1.0, model=_translate(*dot))

        # The sheet is wound several turns, so at the start those turns sit
        # on top of each other. Depth writes plus GL_LESS keep the extra
        # turns from stacking into a solid wall; one glass shell remains.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # While the turns still coincide, writing depth stops them stacking
        # into a solid cone. Once the sheet opens, leave the depth alone so
        # the overlapping paper blends instead of flickering.
        glDepthMask(t <= 1e-4)
        glEnable(GL_CULL_FACE)
        glDepthFunc(GL_LESS)
        glCullFace(GL_FRONT)
        self._draw_mesh(self.sheet, mvp, eye, lit=1.0, alpha=CONE_ALPHA)
        glCullFace(GL_BACK)
        self._draw_mesh(self.sheet, mvp, eye, lit=1.0, alpha=CONE_ALPHA)
        glDepthMask(True)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glDepthFunc(GL_LEQUAL)

        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, fb_w, fb_h)
        w, h = self.window.get_size()
        self.caption.text = caption
        self.caption.x = w * 0.5
        self.caption.y = h - 16
        self.caption.draw()

    def read_rgb(self):
        self.window.switch_to()
        glFinish()
        fb_w, fb_h = self.window.get_framebuffer_size()
        w = fb_w - (fb_w % 2)
        h = fb_h - (fb_h % 2)
        buf = (GLubyte * (w * h * 3))()
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buf)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return np.flipud(frame).copy()

    def present(self):
        self.window.flip()
        self.window.dispatch_events()


def _movie_frames(phase_seconds, fps):
    """Full cycle on the cone, reverse while unrolling, full cycle on the flat net."""
    captions = (
        "Walking around the cone",
        "Walking back while the cone unrolls",
        "Unrolled: the same path is a straight line",
    )
    counts = [max(2, int(round(seconds * fps))) for seconds in phase_seconds]
    for phase, count in enumerate(counts):
        for frame in range(count):
            u = frame / (count - 1)
            if phase == 0:
                t, blend, reverse = 0.0, 0.0, False
            elif phase == 1:
                eased = _smoothstep(u)
                t, blend, reverse = eased, eased, True
            else:
                t, blend, reverse = 1.0, 1.0, False
            walk = 1.0 - u if reverse else u
            index = int(round(walk * (SAMPLES - 1)))
            yield t, index, blend, captions[phase]


def save_animation(path, phase_seconds=(8.0, 6.0, 8.0), fps=30):
    """Record the unwrap movie and return the file path."""
    import imageio.v2 as imageio

    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    frames = list(_movie_frames(phase_seconds, fps))
    print(f"Saving {len(frames)} frames to {path}")
    film = _UnwrapFilm()
    film.open()
    try:
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
        )
        try:
            for number, (t, index, blend, caption) in enumerate(frames, start=1):
                film.draw(t, index, blend, caption)
                writer.append_data(film.read_rgb())
                film.present()
                if number == 1 or number % 30 == 0 or number == len(frames):
                    print(f"  {number}/{len(frames)}  {caption}", flush=True)
        finally:
            writer.close()
    finally:
        film.close()
    print(f"Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Straight line wrapped around a cone.")
    parser.add_argument(
        "-save",
        nargs="?",
        const="cone_unwrap.mp4",
        default=None,
        metavar="FILE",
        help="Save the unwrap animation to FILE (default: cone_unwrap.mp4) and exit.",
    )
    args = parser.parse_args()
    if args.save is not None:
        save_animation(args.save)
    else:
        Demo().run()


if __name__ == "__main__":
    main()
