"""Topology cutting and animated Tutte embedding for triangle meshes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import cg


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class IndexedMesh:
    vertices: FloatArray
    faces: IntArray


@dataclass(slots=True)
class EdgeTopology:
    u: IntArray
    v: IntArray
    counts: IntArray
    first: IntArray
    occurrence_faces: IntArray
    occurrence_locals: IntArray

    @property
    def boundary(self) -> IntArray:
        return np.flatnonzero(self.counts == 1)

    @property
    def shared(self) -> IntArray:
        return np.flatnonzero(self.counts == 2)


@dataclass(slots=True)
class CutDisk:
    vertices: FloatArray
    faces: IntArray
    boundary: IntArray
    used_fallback: bool = False


def index_triangle_soup(points: NDArray[np.floating]) -> IndexedMesh:
    """Weld equal triangle corners and return an indexed triangle mesh."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (3, 3):
        raise ValueError("triangle points must have shape (face_count, 3, 3)")
    vertices, inverse = np.unique(points.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    valid = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    if not np.any(valid):
        raise ValueError("the STL contains no non-degenerate triangles")
    return IndexedMesh(vertices, faces[valid])


def build_edge_topology(faces: IntArray, vertex_count: int) -> EdgeTopology:
    """Return unique undirected edges and their face/local-edge occurrences."""
    face_count = len(faces)
    edge_vertices = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    occurrence_faces = np.tile(np.arange(face_count, dtype=np.int64), 3)
    occurrence_locals = np.repeat(np.arange(3, dtype=np.int64), face_count)
    edge_vertices.sort(axis=1)
    keys = edge_vertices[:, 0] * np.int64(vertex_count) + edge_vertices[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    first = np.flatnonzero(
        np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    ).astype(np.int64)
    counts = np.diff(np.r_[first, len(order)]).astype(np.int64)
    first_occurrences = order[first]
    return EdgeTopology(
        edge_vertices[first_occurrences, 0],
        edge_vertices[first_occurrences, 1],
        counts,
        first,
        occurrence_faces[order],
        occurrence_locals[order],
    )


def _dual_graph(topology: EdgeTopology, face_count: int, edge_ids: IntArray) -> csr_matrix:
    first = topology.first[edge_ids]
    a = topology.occurrence_faces[first]
    b = topology.occurrence_faces[first + 1]
    rows = np.r_[a, b]
    cols = np.r_[b, a]
    data = np.r_[edge_ids + 1, edge_ids + 1]
    return csr_matrix((data, (rows, cols)), shape=(face_count, face_count))


def _compact_mesh(vertices: FloatArray, faces: IntArray) -> IndexedMesh:
    used, inverse = np.unique(faces, return_inverse=True)
    return IndexedMesh(vertices[used], inverse.reshape(-1, 3).astype(np.int64))


def _remove_random_patch(
    mesh: IndexedMesh,
    rng: np.random.Generator,
    patch_fraction: float,
) -> IndexedMesh:
    topology = build_edge_topology(mesh.faces, len(mesh.vertices))
    if np.any(topology.counts > 2):
        raise ValueError("non-manifold edges are not supported")
    dual = _dual_graph(topology, len(mesh.faces), topology.shared)
    if len(mesh.faces) < 2:
        raise ValueError("at least two connected triangles are required")
    target = max(
        1,
        min(2000, len(mesh.faces) - 1, int(len(mesh.faces) * patch_fraction)),
    )

    for _ in range(12):
        seed = int(rng.integers(len(mesh.faces)))
        order = breadth_first_order(dual, seed, directed=False, return_predecessors=False)
        if len(order) <= target:
            continue
        keep = np.ones(len(mesh.faces), dtype=bool)
        keep[order[:target]] = False
        remaining = dual[keep][:, keep]
        components, _ = connected_components(remaining, directed=False)
        if components == 1:
            return _compact_mesh(mesh.vertices, mesh.faces[keep])
    raise ValueError("could not grow a connected random cut patch")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _corner_for_vertex(faces: IntArray, face: int, vertex: int) -> int:
    matches = np.flatnonzero(faces[face] == vertex)
    if len(matches) != 1:
        raise ValueError("invalid triangle corner")
    return face * 3 + int(matches[0])


def _open_along_edges(mesh: IndexedMesh, topology: EdgeTopology, cut: NDArray[np.bool_]) -> CutDisk:
    faces = mesh.faces
    union_find = _UnionFind(len(faces) * 3)
    for edge_id in topology.shared:
        if cut[edge_id]:
            continue
        first = int(topology.first[edge_id])
        face_a = int(topology.occurrence_faces[first])
        face_b = int(topology.occurrence_faces[first + 1])
        u, v = int(topology.u[edge_id]), int(topology.v[edge_id])
        union_find.union(
            _corner_for_vertex(faces, face_a, u),
            _corner_for_vertex(faces, face_b, u),
        )
        union_find.union(
            _corner_for_vertex(faces, face_a, v),
            _corner_for_vertex(faces, face_b, v),
        )

    roots = np.fromiter(
        (union_find.find(i) for i in range(len(faces) * 3)),
        dtype=np.int64,
        count=len(faces) * 3,
    )
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    source_vertices = faces.reshape(-1)[unique_roots]
    disk = CutDisk(
        mesh.vertices[source_vertices],
        inverse.reshape(-1, 3).astype(np.int64),
        np.empty(0, dtype=np.int64),
    )
    disk.boundary = _validate_and_order_boundary(disk)
    return disk


def _validate_and_order_boundary(disk: CutDisk) -> IntArray:
    topology = build_edge_topology(disk.faces, len(disk.vertices))
    if np.any(topology.counts > 2):
        raise ValueError("cut produced non-manifold edges")
    if len(disk.vertices) - len(topology.u) + len(disk.faces) != 1:
        raise ValueError("cut mesh is not a topological disk")

    boundary_edges = topology.boundary
    if len(boundary_edges) < 3:
        raise ValueError("cut mesh has no usable boundary")
    adjacency: dict[int, list[int]] = {}
    for edge_id in boundary_edges:
        u, v = int(topology.u[edge_id]), int(topology.v[edge_id])
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("cut boundary is not one or more simple loops")

    start = min(adjacency)
    boundary = [start]
    previous = -1
    current = start
    while True:
        neighbors = adjacency[current]
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == start:
            break
        if following in boundary:
            raise ValueError("cut boundary intersects itself")
        boundary.append(following)
        previous, current = current, following
    if len(boundary) != len(boundary_edges):
        raise ValueError("cut mesh has multiple boundary loops")
    return np.asarray(boundary, dtype=np.int64)


def _multisource_primal_tree(topology: EdgeTopology, vertex_count: int) -> tuple[IntArray, IntArray]:
    boundary_edges = topology.boundary
    roots = np.unique(np.r_[topology.u[boundary_edges], topology.v[boundary_edges]])
    if len(roots) == 0:
        raise ValueError("a patch boundary is required before topology cutting")

    edge_ids = np.arange(len(topology.u), dtype=np.int64)
    rows = np.r_[topology.u, topology.v]
    cols = np.r_[topology.v, topology.u]
    data = np.r_[edge_ids + 1, edge_ids + 1]
    graph = csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))

    parent = np.full(vertex_count, -2, dtype=np.int64)
    parent_edge = np.full(vertex_count, -1, dtype=np.int64)
    queue: deque[int] = deque(int(root) for root in roots)
    parent[roots] = -1
    while queue:
        vertex = queue.popleft()
        begin, end = graph.indptr[vertex], graph.indptr[vertex + 1]
        for index in range(begin, end):
            neighbor = int(graph.indices[index])
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = vertex
            parent_edge[neighbor] = int(graph.data[index]) - 1
            queue.append(neighbor)
    if np.any(parent == -2):
        raise ValueError("mesh vertex graph is disconnected")
    return parent, parent_edge


def _tree_cotree_disk(mesh: IndexedMesh) -> CutDisk:
    topology = build_edge_topology(mesh.faces, len(mesh.vertices))
    if np.any(topology.counts > 2):
        raise ValueError("mesh has non-manifold edges")
    parent, parent_edge = _multisource_primal_tree(topology, len(mesh.vertices))
    primal_tree = np.zeros(len(topology.u), dtype=bool)
    primal_tree[parent_edge[parent_edge >= 0]] = True

    dual_candidates = topology.shared[~primal_tree[topology.shared]]
    dual = _dual_graph(topology, len(mesh.faces), dual_candidates)
    order, predecessors = breadth_first_order(
        dual, 0, directed=False, return_predecessors=True
    )
    if len(order) != len(mesh.faces):
        raise ValueError("tree-cotree dual graph is disconnected")

    dual_tree = np.zeros(len(topology.u), dtype=bool)
    children = order[1:]
    parents = predecessors[children]
    edge_values = np.asarray(dual[children, parents]).reshape(-1)
    dual_tree[edge_values.astype(np.int64) - 1] = True

    generators = topology.shared[
        ~primal_tree[topology.shared] & ~dual_tree[topology.shared]
    ]
    cut = np.zeros(len(topology.u), dtype=bool)
    cut[topology.boundary] = True
    cut[generators] = True

    for edge_id in generators:
        for endpoint in (int(topology.u[edge_id]), int(topology.v[edge_id])):
            vertex = endpoint
            while parent[vertex] >= 0:
                path_edge = int(parent_edge[vertex])
                cut[path_edge] = True
                vertex = int(parent[vertex])
    return _open_along_edges(mesh, topology, cut)


def _dual_tree_disk(mesh: IndexedMesh) -> CutDisk:
    """Robust fallback: retain only dual-tree face gluings."""
    topology = build_edge_topology(mesh.faces, len(mesh.vertices))
    dual = _dual_graph(topology, len(mesh.faces), topology.shared)
    order, predecessors = breadth_first_order(
        dual, 0, directed=False, return_predecessors=True
    )
    if len(order) != len(mesh.faces):
        raise ValueError("remaining surface is disconnected")
    keep_glued = np.zeros(len(topology.u), dtype=bool)
    children = order[1:]
    values = np.asarray(dual[children, predecessors[children]]).reshape(-1)
    keep_glued[values.astype(np.int64) - 1] = True
    cut = ~keep_glued
    disk = _open_along_edges(mesh, topology, cut)
    disk.used_fallback = True
    return disk


def make_random_cut_disk(
    mesh: IndexedMesh,
    *,
    rng: np.random.Generator | None = None,
    patch_fraction: float = 0.005,
) -> CutDisk:
    """Remove a random patch and cut any remaining topology into one disk."""
    rng = rng or np.random.default_rng()
    opened = _remove_random_patch(mesh, rng, patch_fraction)
    try:
        return _tree_cotree_disk(opened)
    except ValueError:
        return _dual_tree_disk(opened)


def circular_boundary_positions(disk: CutDisk) -> FloatArray:
    boundary = disk.boundary
    following = np.roll(boundary, -1)
    lengths = np.linalg.norm(
        disk.vertices[following] - disk.vertices[boundary], axis=1
    )
    lengths = np.maximum(lengths, np.finfo(np.float64).eps)
    angles = 2.0 * np.pi * np.r_[0.0, np.cumsum(lengths[:-1])] / lengths.sum()
    return np.column_stack((np.cos(angles), np.sin(angles)))


class SpringEmbedding:
    """Animated damped spring relaxation toward a Tutte equilibrium."""

    def __init__(self, disk: CutDisk, rng: np.random.Generator | None = None) -> None:
        self.disk = disk
        self.rng = rng or np.random.default_rng()
        self.boundary_positions = circular_boundary_positions(disk)
        topology = build_edge_topology(disk.faces, len(disk.vertices))
        rows = np.r_[topology.u, topology.v]
        cols = np.r_[topology.v, topology.u]
        adjacency = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(disk.vertices), len(disk.vertices)),
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        self.average = adjacency.multiply((1.0 / degree)[:, None]).tocsr()
        self.laplacian = diags(degree, format="csr") - adjacency
        self.boundary_mask = np.zeros(len(disk.vertices), dtype=bool)
        self.boundary_mask[disk.boundary] = True
        self.interior = np.flatnonzero(~self.boundary_mask)

        centered = disk.vertices - disk.vertices.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered))
        _, axes = np.linalg.eigh(covariance)
        projected = centered @ axes[:, -2:]
        radius = np.linalg.norm(projected, axis=1).max(initial=1.0)
        normalized_projection = projected / max(radius, 1e-12)
        # Duplicate seam vertices begin at the same projected 3D location.
        # The viewer animates them spreading into the circular cut disk.
        self.flatten_start_positions = normalized_projection * 0.72
        self.positions = normalized_projection * 0.45
        self.positions += self.rng.normal(0.0, 0.015, self.positions.shape)
        self.positions[disk.boundary] = self.boundary_positions
        self.velocity = np.zeros_like(self.positions)
        self.target = self._solve_equilibrium()
        self.energy = float("inf")
        self.settled = len(self.interior) == 0
        self.elapsed = 0.0

    def _solve_equilibrium(self) -> FloatArray:
        target = np.zeros_like(self.positions)
        target[self.disk.boundary] = self.boundary_positions
        if len(self.interior) == 0:
            return target
        system = self.laplacian[self.interior][:, self.interior]
        boundary_term = (
            -self.laplacian[self.interior][:, self.disk.boundary]
            @ self.boundary_positions
        )
        for axis in range(2):
            solution, status = cg(
                system,
                np.asarray(boundary_term[:, axis]).ravel(),
                rtol=1e-8,
                atol=1e-10,
                maxiter=5000,
            )
            if status != 0:
                raise ValueError(f"Laplacian equilibrium solve failed ({status})")
            target[self.interior, axis] = solution
        return target

    def step(self, dt: float) -> None:
        if self.settled:
            return
        dt = min(max(dt, 0.0), 1.0 / 20.0)
        spring_force = self.average @ self.positions - self.positions
        target_force = self.target - self.positions
        acceleration = (
            18.0 * spring_force + 6.0 * target_force - 2.0 * self.velocity
        )
        acceleration[self.boundary_mask] = 0.0
        self.velocity += acceleration * dt
        self.positions += self.velocity * dt
        self.positions[self.disk.boundary] = self.boundary_positions
        self.velocity[self.boundary_mask] = 0.0
        self.elapsed += dt
        difference = self.target - self.positions
        self.energy = float(np.sqrt(np.mean(difference[self.interior] ** 2)))
        speed = float(np.max(np.linalg.norm(self.velocity[self.interior], axis=1), initial=0.0))
        if (
            self.elapsed > 2.5
            and self.energy < 2e-4
            and speed < 2e-3
        ) or self.elapsed > 14.0:
            self.positions[:] = self.target
            self.velocity.fill(0.0)
            self.energy = 0.0
            self.settled = True
