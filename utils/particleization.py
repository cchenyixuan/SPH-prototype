from typing import Iterable
from collections import deque

import numpy as np
from half_edge_mesh import HalfEdgeMesh, HalfEdgeVertex, HalfEdgeFacet, HalfEdge


class Particleization:
    """
    This class provides methods to discretize geometry into particles.
    """

    def __init__(self, half_edge_mesh: HalfEdgeMesh, particle_radius: float):
        self.half_edge_mesh: HalfEdgeMesh = half_edge_mesh
        self.particle_radius: float = particle_radius
        self.edge_particles: set = set()
        self.facet_particles: set = set()

    def discretize_edges(self):
        # edge_particles
        todo_list = set(self.half_edge_mesh.half_edges.keys())
        # for all edges, discrete into particles
        for half_edge_key in self.half_edge_mesh.half_edges.keys():

            half_edge = self.half_edge_mesh.half_edges[half_edge_key]
            # If this edge has been checked
            if half_edge.index not in todo_list:
                continue
            # shorten the edge until nomore new particle can be added
            particle1 = np.array(half_edge.vertex.numpy)
            particle2 = np.array(half_edge.pair.vertex.numpy)
            distance = np.linalg.norm(particle2 - particle1)
            direction = (particle2 - particle1) / distance
            while True:

                self.edge_particles.add((particle1[0], particle1[1], particle1[2]))
                particle1 += direction * 2 * self.particle_radius
                distance -= 2 * self.particle_radius
                if distance < 1.5 * self.particle_radius:
                    break
            # remove from checklist
            todo_list.remove(half_edge.index)
            if half_edge.pair is not None:
                todo_list.remove(half_edge.pair.index)

    def discretize_facets(self):
        for facet_key in self.half_edge_mesh.half_edge_facets.keys():
            facet = self.half_edge_mesh.half_edge_facets[facet_key]
            nodes = [half_edge.vertex.numpy for half_edge in facet.half_edge.values()]

            p1, p2, p3 = nodes[0], nodes[1], nodes[2]
            center = 1 / 3 * (p1 + p2 + p3)
            up = (p1 - center) / np.linalg.norm(p1 - center)
            down = -up
            left = np.cross(facet.cal_normal(), up) / np.linalg.norm(np.cross(facet.cal_normal(), up))
            right = -left
            # shrink 2 radius
            p1 = p1 + (center - p1) / np.linalg.norm(center - p1) * self.particle_radius * 1.1
            p2 = p2 + (center - p2) / np.linalg.norm(center - p2) * self.particle_radius * 1.1
            p3 = p3 + (center - p3) / np.linalg.norm(center - p3) * self.particle_radius * 1.1
            # iterative part
            todo_points = deque()
            todo_points.append((0, 0))
            checked_points = set()
            while todo_points:
                x_coord, y_coord = todo_points.pop()
                point = center + right * x_coord * 2 * self.particle_radius + up * y_coord * 2 * self.particle_radius
                if self.point_inside_facet(p1, p2, p3, point):
                    self.facet_particles.add((point[0], point[1], point[2]))
                    todo_points.append((x_coord + 1, y_coord)) if (x_coord + 1, y_coord) not in checked_points else None
                    todo_points.append((x_coord - 1, y_coord)) if (x_coord - 1, y_coord) not in checked_points else None
                    todo_points.append((x_coord, y_coord + 1)) if (x_coord, y_coord + 1) not in checked_points else None
                    todo_points.append((x_coord, y_coord - 1)) if (x_coord, y_coord - 1) not in checked_points else None
                checked_points.add((x_coord, y_coord))

    def extend(self, layers=1, reverse=False):
        """
        Extend the geometry along with the normal vectors,
        :param layers:
        :param reverse:
        :return:
        """
        ...

    @staticmethod
    def point_inside_facet(p1, p2, p3, point: np.ndarray) -> bool:
        """
        Check if a point is inside the facet, only valid for point within the plane
        """
        tmp = np.cross(p2 - p1, p3 - p2)
        normal = tmp / np.linalg.norm(tmp)
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if 0.5 * np.cross(p1 - point, p2 - point) @ normal / area > 0 and np.cross(p2 - point,
                                                                                   p3 - point) @ normal / area > 0 and np.cross(
                p3 - point, p1 - point) @ normal / area > 0:
            return True
        else:
            return False


if __name__ == '__main__':
    test_mesh = r"untitled.obj"
    v, fac = HalfEdgeMesh.load_obj(test_mesh)
    particle_system = Particleization(HalfEdgeMesh(v, fac), 0.005)
    particle_system.discretize_edges()
    print(particle_system.edge_particles)
    particle_system.discretize_facets()
    print(particle_system.facet_particles)
    with open("o.obj", "w") as f:
        for item in particle_system.edge_particles:
            f.write(f"v {item[0]} {item[1]} {item[2]}\n")
        for item in particle_system.facet_particles:
            f.write(f"v {item[0]} {item[1]} {item[2]}\n")
        f.close()
