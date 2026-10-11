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
    GL_TRIANGLES,
    glClear,
    glClearColor,
    glViewport,
)
try:
    from pyglet.gl.gl_compat import (
        GL_AMBIENT, GL_DIFFUSE, GL_FILL, GL_FRONT_AND_BACK, GL_LIGHT0,
        GL_LIGHTING, GL_LINE, GL_MODELVIEW, GL_NORMALIZE, GL_POSITION,
        GL_PROJECTION, GL_SMOOTH, GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_FLOAT,
        GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHT_MODEL_AMBIENT, GL_TRUE, GL_EMISSION,
        GLfloat, glEnable, glEnableClientState, glDisableClientState,
        glVertexPointer, glNormalPointer, glDrawArrays, glLightfv,
        glLightModeli, glLightModelfv,
        glLoadIdentity, glMaterialfv, glMatrixMode, glPolygonMode, glFrustum,
        glRotatef, glShadeModel, glTranslatef,
    )
except ImportError:  # Pyglet 1.x
    from pyglet.gl import (
        GL_AMBIENT, GL_DIFFUSE, GL_FILL, GL_FRONT_AND_BACK, GL_LIGHT0,
        GL_LIGHTING, GL_LINE, GL_MODELVIEW, GL_NORMALIZE, GL_POSITION,
        GL_PROJECTION, GL_SMOOTH, GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_FLOAT,
        GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHT_MODEL_AMBIENT, GL_TRUE, GL_EMISSION,
        GLfloat, glEnable, glEnableClientState, glDisableClientState,
        glVertexPointer, glNormalPointer, glDrawArrays, glLightfv,
        glLightModeli, glLightModelfv,
        glLoadIdentity, glMaterialfv, glMatrixMode, glPolygonMode, glFrustum,
        glRotatef, glShadeModel, glTranslatef,
    )
from pyglet.window import key, mouse


Vec3 = tuple[float, float, float]


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
        self.positions = (GLfloat * len(positions))(*positions)
        self.normals = (GLfloat * len(normals))(*normals)

        self.rot_x = 20.0
        self.rot_y = -30.0
        self.pan_x = self.pan_y = 0.0
        self.distance = self.model_size * 2.8
        self.wireframe = False

        glClearColor(0.08, 0.09, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        # STL normals are often reversed or inconsistent. Two-sided lighting
        # plus a bright ambient component keeps either side clearly visible.
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelfv(
            GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(0.35, 0.35, 0.35, 1.0)
        )
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.40, 0.40, 0.40, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1.0, 1.0, 1.0, 1.0))
        glMaterialfv(
            GL_FRONT_AND_BACK, GL_AMBIENT, (GLfloat * 4)(0.35, 0.65, 0.90, 1.0)
        )
        glMaterialfv(
            GL_FRONT_AND_BACK, GL_DIFFUSE, (GLfloat * 4)(0.35, 0.70, 1.00, 1.0)
        )
        glMaterialfv(
            GL_FRONT_AND_BACK, GL_EMISSION, (GLfloat * 4)(0.04, 0.08, 0.12, 1.0)
        )

    def _reset(self) -> None:
        self.rot_x, self.rot_y = 20.0, -30.0
        self.pan_x = self.pan_y = 0.0
        self.distance = self.model_size * 2.8

    def on_resize(self, width: int, height: int):
        glViewport(0, 0, width, max(1, height))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        near, far = self.model_size / 1000, self.model_size * 1000
        top = near * math.tan(math.radians(45.0) / 2)
        right = top * width / max(1, height)
        glFrustum(-right, right, -top, top, near, far)
        glMatrixMode(GL_MODELVIEW)
        return super().on_resize(width, height)

    def on_draw(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, -self.distance)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(1.0, 1.0, 2.0, 0.0))
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.positions)
        glNormalPointer(GL_FLOAT, 0, self.normals)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

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
