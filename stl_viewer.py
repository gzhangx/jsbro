"""An interactive STL viewer and animated spring flattening demo.

Usage:
    pip install pyglet numpy scipy
    python stl_viewer.py path/to/model.stl

Controls:
    Button          Randomly cut and flatten / return to 3D
    Left drag       Rotate
    Right drag      Pan
    Mouse wheel     Zoom
    R               Reset the view
    W               Toggle wireframe
    Escape          Close
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import math
import struct
from pathlib import Path

import numpy as np
import pyglet
from pyglet.gl import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LINE,
    GL_LINE_LOOP,
    GL_TRIANGLES,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glLineWidth,
    glPolygonMode,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3 as PygletVec3
from pyglet.window import key, mouse

from mesh_flatten import (
    IndexedMesh,
    SpringEmbedding,
    index_triangle_soup,
    make_random_cut_disk,
)


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


FLAT_VERTEX_SHADER = """#version 330 core
in vec2 position;
uniform vec2 viewport_scale;

void main()
{
    gl_Position = vec4(position * viewport_scale, 0.0, 1.0);
}
"""


FLAT_FRAGMENT_SHADER = """#version 330 core
uniform vec4 color;
out vec4 final_color;

void main()
{
    final_color = color;
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
        triangle_points = np.asarray(
            [(a, b, c) for _, a, b, c in triangles], dtype=np.float64
        )
        self.indexed_mesh = index_triangle_soup(triangle_points)
        flat_points = triangle_points.reshape(-1, 3)
        mins = flat_points.min(axis=0)
        maxs = flat_points.max(axis=0)
        self.center = tuple((mins + maxs) / 2)
        self.model_size = float(np.max(maxs - mins)) or 1.0
        positions = flat_points.astype(np.float32, copy=False).reshape(-1)
        face_normals = np.asarray([normal for normal, *_ in triangles], dtype=np.float32)
        normals = np.repeat(face_normals, 3, axis=0).reshape(-1)
        self.vertex_count = len(flat_points)
        self.program = ShaderProgram(
            Shader(VERTEX_SHADER, "vertex"),
            Shader(FRAGMENT_SHADER, "fragment"),
        )
        self.flat_program = ShaderProgram(
            Shader(FLAT_VERTEX_SHADER, "vertex"),
            Shader(FLAT_FRAGMENT_SHADER, "fragment"),
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
        self.mode = "3d"
        self.solver: SpringEmbedding | None = None
        self.flat_vertex_list = None
        self.boundary_vertex_list = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-cut")
        self.flatten_future: Future[SpringEmbedding] | None = None
        self.status_message = ""
        self.projection_matrix = Mat4.perspective_projection(
            self.width / max(1, self.height),
            self.model_size / 1000,
            self.model_size * 1000,
            fov=45.0,
        )

        glClearColor(0.08, 0.09, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.button = pyglet.shapes.Rectangle(
            20, self.height - 58, 235, 38, color=(42, 112, 178)
        )
        self.button_label = pyglet.text.Label(
            "Random Cut & Flatten",
            x=self.button.x + self.button.width / 2,
            y=self.button.y + self.button.height / 2,
            anchor_x="center",
            anchor_y="center",
            font_size=11,
        )
        self.status_label = pyglet.text.Label(
            "",
            x=20,
            y=self.height - 78,
            anchor_x="left",
            anchor_y="top",
            font_size=10,
            color=(220, 225, 235, 255),
        )
        pyglet.clock.schedule_interval(self._update, 1.0 / 60.0)

    @staticmethod
    def _prepare_flattening(mesh: IndexedMesh, seed: int) -> SpringEmbedding:
        rng = np.random.default_rng(seed)
        disk = make_random_cut_disk(mesh, rng=rng)
        return SpringEmbedding(disk, rng)

    def _start_flattening(self) -> None:
        if self.mode != "3d":
            return
        self.mode = "building"
        self.status_message = "Building a random topology cut..."
        self.button.color = (75, 80, 92)
        seed = int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))
        self.flatten_future = self.executor.submit(
            self._prepare_flattening, self.indexed_mesh, seed
        )

    def _create_flat_buffers(self, solver: SpringEmbedding) -> None:
        positions = solver.positions.astype(np.float32, copy=False).reshape(-1)
        indices = solver.disk.faces.astype(np.uint32, copy=False).reshape(-1)
        self.flat_vertex_list = self.flat_program.vertex_list_indexed(
            len(solver.positions),
            GL_TRIANGLES,
            indices,
            position=("f", positions),
        )
        self.boundary_vertex_list = self.flat_program.vertex_list_indexed(
            len(solver.positions),
            GL_LINE_LOOP,
            solver.disk.boundary.astype(np.uint32, copy=False),
            position=("f", positions),
        )

    def _sync_flat_positions(self) -> None:
        if self.solver is None or self.flat_vertex_list is None:
            return
        positions = self.solver.positions.astype(np.float32, copy=False).reshape(-1)
        self.flat_vertex_list.position[:] = positions
        self.boundary_vertex_list.position[:] = positions

    def _return_to_3d(self) -> None:
        self.mode = "3d"
        self.solver = None
        self.flat_vertex_list = None
        self.boundary_vertex_list = None
        self.status_message = ""
        self.button.color = (42, 112, 178)
        self.button_label.text = "Random Cut & Flatten"

    def _update(self, dt: float) -> None:
        if self.mode == "building" and self.flatten_future is not None:
            if self.flatten_future.done():
                try:
                    self.solver = self.flatten_future.result()
                    self._create_flat_buffers(self.solver)
                except Exception as exc:
                    self.mode = "3d"
                    self.button.color = (42, 112, 178)
                    self.status_message = f"Could not flatten mesh: {exc}"
                else:
                    self.mode = "relaxing"
                    self.button.color = (150, 70, 55)
                    self.button_label.text = "Back to 3D"
                    fallback = " (fallback seams)" if self.solver.disk.used_fallback else ""
                    self.status_message = f"Springs relaxing{fallback}..."
                finally:
                    self.flatten_future = None
        elif self.mode == "relaxing" and self.solver is not None:
            self.solver.step(dt)
            self._sync_flat_positions()
            self.status_message = f"Spring error: {self.solver.energy:.6f}"
            if self.solver.settled:
                self.mode = "flat"
                self.status_message = "Laplacian equilibrium settled"
        self.status_label.text = self.status_message

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
        if hasattr(self, "button"):
            self.button.y = height - 58
            self.button_label.x = self.button.x + self.button.width / 2
            self.button_label.y = self.button.y + self.button.height / 2
            self.status_label.y = height - 78
        return result

    def on_draw(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.mode in {"relaxing", "flat"}:
            self._draw_flat()
        else:
            self._draw_3d()
        glDisable(GL_DEPTH_TEST)
        self.button.draw()
        self.button_label.draw()
        self.status_label.draw()

    def _draw_3d(self) -> None:
        glEnable(GL_DEPTH_TEST)
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

    def _draw_flat(self) -> None:
        if self.flat_vertex_list is None or self.boundary_vertex_list is None:
            return
        glDisable(GL_DEPTH_TEST)
        shortest = max(1, min(self.width, self.height))
        scale = (
            0.84 * shortest / max(1, self.width),
            0.84 * shortest / max(1, self.height),
        )
        self.flat_program.use()
        self.flat_program["viewport_scale"] = scale
        self.flat_program["color"] = (0.07, 0.25, 0.42, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.flat_vertex_list.draw(GL_TRIANGLES)
        self.flat_program["color"] = (0.30, 0.72, 1.0, 1.0)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.flat_vertex_list.draw(GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.flat_program["color"] = (1.0, 0.58, 0.18, 1.0)
        glLineWidth(3.0)
        self.boundary_vertex_list.draw(GL_LINE_LOOP)
        glLineWidth(1.0)
        self.flat_program.stop()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if self.mode != "3d":
            return
        if buttons & mouse.LEFT:
            self.rot_y += dx * 0.5
            self.rot_x -= dy * 0.5
        if buttons & mouse.RIGHT:
            scale = self.distance / max(self.width, self.height)
            self.pan_x += dx * scale
            self.pan_y += dy * scale

    def on_mouse_press(self, x, y, button, modifiers) -> None:
        if button != mouse.LEFT:
            return
        if (
            self.button.x <= x <= self.button.x + self.button.width
            and self.button.y <= y <= self.button.y + self.button.height
        ):
            if self.mode == "3d":
                self._start_flattening()
            elif self.mode in {"relaxing", "flat"}:
                self._return_to_3d()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        if self.mode == "3d":
            self.distance = max(self.model_size * 0.05, self.distance * (0.9**scroll_y))

    def on_key_press(self, symbol, modifiers) -> None:
        if symbol == key.R and self.mode == "3d":
            self._reset()
        elif symbol == key.W and self.mode == "3d":
            self.wireframe = not self.wireframe
        elif symbol == key.ESCAPE:
            self.close()

    def on_close(self) -> None:
        pyglet.clock.unschedule(self._update)
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().on_close()


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
