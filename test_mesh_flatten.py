import unittest

import numpy as np

from mesh_flatten import (
    SpringEmbedding,
    build_edge_topology,
    circular_boundary_positions,
    index_triangle_soup,
    make_random_cut_disk,
)


def sphere_triangle_soup(latitude_segments: int = 10, longitude_segments: int = 20):
    vertices = [(0.0, 0.0, 1.0)]
    for latitude in range(1, latitude_segments):
        phi = np.pi * latitude / latitude_segments
        for longitude in range(longitude_segments):
            theta = 2.0 * np.pi * longitude / longitude_segments
            vertices.append(
                (
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                )
            )
    south = len(vertices)
    vertices.append((0.0, 0.0, -1.0))

    def ring(latitude: int, longitude: int) -> int:
        return 1 + (latitude - 1) * longitude_segments + longitude % longitude_segments

    faces = []
    for longitude in range(longitude_segments):
        following = (longitude + 1) % longitude_segments
        faces.append((0, ring(1, following), ring(1, longitude)))
    for latitude in range(1, latitude_segments - 1):
        for longitude in range(longitude_segments):
            following = (longitude + 1) % longitude_segments
            a, b = ring(latitude, longitude), ring(latitude, following)
            c, d = ring(latitude + 1, longitude), ring(latitude + 1, following)
            faces.extend(((a, b, c), (b, d, c)))
    for longitude in range(longitude_segments):
        following = (longitude + 1) % longitude_segments
        faces.append(
            (
                ring(latitude_segments - 1, longitude),
                ring(latitude_segments - 1, following),
                south,
            )
        )
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices[np.asarray(faces, dtype=np.int64)]


class MeshFlattenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = index_triangle_soup(sphere_triangle_soup())
        cls.disk = make_random_cut_disk(
            cls.mesh, rng=np.random.default_rng(1234), patch_fraction=0.08
        )

    def test_triangle_soup_is_welded(self):
        self.assertEqual(len(self.mesh.vertices), 182)
        self.assertEqual(len(self.mesh.faces), 360)
        self.assertTrue(np.all(self.mesh.faces[:, 0] != self.mesh.faces[:, 1]))

    def test_random_cut_is_one_disk(self):
        topology = build_edge_topology(self.disk.faces, len(self.disk.vertices))
        euler = len(self.disk.vertices) - len(topology.u) + len(self.disk.faces)
        self.assertEqual(euler, 1)
        self.assertEqual(len(topology.boundary), len(self.disk.boundary))
        self.assertGreater(len(self.disk.boundary), 2)

    def test_boundary_maps_to_unit_circle(self):
        positions = circular_boundary_positions(self.disk)
        radii = np.linalg.norm(positions, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(positions)))

    def test_spring_embedding_converges(self):
        solver = SpringEmbedding(self.disk, np.random.default_rng(5))
        self.assertTrue(np.all(np.isfinite(solver.target)))
        for _ in range(600):
            solver.step(1.0 / 60.0)
        self.assertTrue(solver.settled)
        np.testing.assert_allclose(solver.positions, solver.target, atol=1e-12)
        np.testing.assert_allclose(
            solver.positions[self.disk.boundary],
            solver.boundary_positions,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
