"""A small, dependency-light STL viewer built with pyglet.

Usage:
    pip install pyglet
    python stl_viewer.py path/to/model.stl

Controls:
    Left drag       Rotate
    Right drag      Pan
    Mouse wheel     Zoom
    R               Reset the view
    W               Toggle wireframe
    Escape          Close
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import pyglet
from pyglet.gl import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_TRIANGLES,
    glClear,
    glClearColor,
    glEnable,
    glPolygonMode,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3 as PygletVec3
from pyglet.window import key, mouse


Vec3 = tuple[float, float, float]


VERTEX_SHADER = """#version 330 core
in vec3 position;
in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 surface_normal;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    surface_normal = mat3(transpose(inverse(model))) * normal;
}
"""


FRAGMENT_SHADER = """#version 330 core
in vec3 surface_normal;
out vec4 final_color;

void main()
{
    vec3 n = normalize(surface_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 light_direction = normalize(vec3(0.4, 0.6, 1.0));
    float brightness = 0.30 + 0.70 * abs(dot(n, light_direction));
    vec3 blue = vec3(0.28, 0.68, 1.0);
    final_color = vec4(blue * brightness, 1.0);
}
"""


def _normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (0.0, 0.0, 1.0) if length == 0 else (nx / length, ny / length, nz / length)


def _read_binary(data: bytes) -> list[tuple[Vec3, Vec3, Vec3, Vec3]]:
    count = struct.unpack_from("<I", data, 80)[0]
    triangles = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        supplied_normal = values[0:3]
        a, b, c = values[3:6], values[6:9], values[9:12]
        normal = supplied_normal if any(supplied_normal) else _normal(a, b, c)
        triangles.append((normal, a, b, c))
        offset += 50
    return triangles


def _read_ascii(text: str) -> list[tuple[Vec3, Vec3, Vec3, Vec3]]:
    triangles = []
    facet_normal: Vec3 | None = None
    vertices: list[Vec3] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].lower() == "facet" and parts[1].lower() == "normal":
            facet_normal = tuple(map(float, parts[2:5]))  # type: ignore[assignment]
        elif len(parts) >= 4 and parts[0].lower() == "vertex":
            vertices.append(tuple(map(float, parts[1:4])))  # type: ignore[arg-type]
            if len(vertices) == 3:
                a, b, c = vertices
                normal = facet_normal if facet_normal and any(facet_normal) else _normal(a, b, c)
                triangles.append((normal, a, b, c))
                vertices = []
    if not triangles:
        raise ValueError("No triangles were found in the ASCII STL")
    return triangles


def load_stl(path: Path) -> list[tuple[Vec3, Vec3, Vec3, Vec3]]:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            return _read_binary(data)
    try:
        return _read_ascii(data.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid or unsupported STL file") from exc


class STLViewer(pyglet.window.Window):
    def __init__(self, path: Path) -> None:
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(900, 700, f"STL Viewer - {path.name}", resizable=True, config=config)

        triangles = load_stl(path)
        points = [point for _, a, b, c in triangles for point in (a, b, c)]
        mins = tuple(min(p[i] for p in points) for i in range(3))
        maxs = tuple(max(p[i] for p in points) for i in range(3))
        self.center = tuple((mins[i] + maxs[i]) / 2 for i in range(3))
        self.model_size = max(maxs[i] - mins[i] for i in range(3)) or 1.0

        positions: list[float] = []
        normals: list[float] = []
        for normal, a, b, c in triangles:
            positions.extend((*a, *b, *c))
            normals.extend(normal * 3)
        self.vertex_count = len(points)
        self.program = ShaderProgram(
            Shader(VERTEX_SHADER, "vertex"),
            Shader(FRAGMENT_SHADER, "fragment"),
        )
        self.vertex_list = self.program.vertex_list(
            self.vertex_count,
            GL_TRIANGLES,
            position=("f", positions),
            normal=("f", normals),
        )

        self.rot_x = 20.0
        self.rot_y = -30.0
        self.pan_x = self.pan_y = 0.0
        self.distance = self.model_size * 2.8
        self.wireframe = False
        self.projection_matrix = Mat4.perspective_projection(
            self.width / max(1, self.height),
            self.model_size / 1000,
            self.model_size * 1000,
            fov=45.0,
        )

        glClearColor(0.08, 0.09, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

    def _reset(self) -> None:
        self.rot_x, self.rot_y = 20.0, -30.0
        self.pan_x = self.pan_y = 0.0
        self.distance = self.model_size * 2.8

    def on_resize(self, width: int, height: int):
        result = super().on_resize(width, height)
        glViewport(0, 0, width, max(1, height))
        near, far = self.model_size / 1000, self.model_size * 1000
        self.projection_matrix = Mat4.perspective_projection(
            width / max(1, height), near, far, fov=45.0
        )
        return result

    def on_draw(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        center = Mat4.from_translation(
            PygletVec3(-self.center[0], -self.center[1], -self.center[2])
        )
        rotation = Mat4.from_rotation(
            math.radians(self.rot_x), PygletVec3(1.0, 0.0, 0.0)
        ) @ Mat4.from_rotation(
            math.radians(self.rot_y), PygletVec3(0.0, 1.0, 0.0)
        )
        view = Mat4.from_translation(
            PygletVec3(self.pan_x, self.pan_y, -self.distance)
        )
        self.program.use()
        self.program["model"] = rotation @ center
        self.program["view"] = view
        self.program["projection"] = self.projection_matrix
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        self.vertex_list.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.program.stop()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if buttons & mouse.LEFT:
            self.rot_y += dx * 0.5
            self.rot_x -= dy * 0.5
        if buttons & mouse.RIGHT:
            scale = self.distance / max(self.width, self.height)
            self.pan_x += dx * scale
            self.pan_y += dy * scale

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        self.distance = max(self.model_size * 0.05, self.distance * (0.9**scroll_y))

    def on_key_press(self, symbol, modifiers) -> None:
        if symbol == key.R:
            self._reset()
        elif symbol == key.W:
            self.wireframe = not self.wireframe
        elif symbol == key.ESCAPE:
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display an ASCII or binary STL file with pyglet.")
    parser.add_argument("stl_file", type=Path, help="path to the STL file")
    args = parser.parse_args()
    if not args.stl_file.is_file():
        parser.error(f"file does not exist: {args.stl_file}")
    STLViewer(args.stl_file)
    pyglet.app.run()


if __name__ == "__main__":
    main()
