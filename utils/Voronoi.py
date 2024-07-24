import numpy as np
import random
from collections import deque
from half_edge_mesh import HalfEdge, HalfEdgeVertex, HalfEdgeFacet, HalfEdgeMesh


class VoronoiNode:
    def __init__(self, position: tuple, index):
        self.position = position
        self.numpy = np.array(position, dtype=np.int32)
        self.index = index
        self.count = 1
        self.territory = set()
        self.territory.add(position)

    def move(self, position):
        self.position = position
        self.numpy = np.array(position, dtype=np.int32)
        self.count = 1
        self.territory = set()
        self.territory.add(position)


class Voronoi:
    def __init__(self, radius: float, half_edge_mesh: HalfEdgeMesh | None = None):
        self.mesh = half_edge_mesh
        self.radius = radius
        self.pixel_factor = 0.25
        self.available_lattices = None

    def generation(self, facet: HalfEdgeFacet, seeds=None):
        """
        Jump Flood 'https://www.bilibili.com/video/BV1rN4y1V7GR/?vd_source=87e7363543cf6216fb81deb48e356178'
        :return:
        """
        # first we rasterize our triangle into lattices with edge-length = 0.1 * radius
        facet_lattices = set()
        nodes = [half_edge.vertex.numpy for half_edge in facet.half_edge.values()]
        p1, p2, p3 = nodes[0], nodes[1], nodes[2]
        center = 1 / 3 * (p1 + p2 + p3)
        up = (p1 - center) / np.linalg.norm(p1 - center)
        down = -up
        left = np.cross(facet.cal_normal(), up) / np.linalg.norm(np.cross(facet.cal_normal(), up))
        right = -left

        up = np.array((0, 1, 0.0))
        right = np.array((1, 0.0, 0.0))

        # iterative part
        todo_points = deque()
        todo_points.append((0, 0))
        checked_points = set()
        while todo_points:
            x_coord, y_coord = todo_points.pop()
            point = center + right * x_coord * self.radius * self.pixel_factor + up * y_coord * self.radius * self.pixel_factor
            if self.point_inside_facet(p1, p2, p3, point):
                facet_lattices.add((x_coord, y_coord))
                todo_points.append((x_coord + 1, y_coord)) if (x_coord + 1, y_coord) not in checked_points else None
                todo_points.append((x_coord - 1, y_coord)) if (x_coord - 1, y_coord) not in checked_points else None
                todo_points.append((x_coord, y_coord + 1)) if (x_coord, y_coord + 1) not in checked_points else None
                todo_points.append((x_coord, y_coord - 1)) if (x_coord, y_coord - 1) not in checked_points else None
            checked_points.add((x_coord, y_coord))

        # we randomly generate particles inside the lattices if no seeds
        if seeds is None:
            voronoi_node_count = int(facet.cal_area() // (np.pi * self.radius ** 2) + 1)
            voronoi_node_seeds = random.sample(sorted(facet_lattices), voronoi_node_count)
            voronoi_nodes = [VoronoiNode(position, index) for index, position in enumerate(voronoi_node_seeds)]
            available_lattices = {lattice: -1 for lattice in facet_lattices}
            for voronoi_node in voronoi_nodes:
                available_lattices[voronoi_node.position] = voronoi_node.index
        else:
            voronoi_node_count = len(seeds)
            voronoi_nodes = seeds
            available_lattices = {lattice: -1 for lattice in facet_lattices}
            for voronoi_node in voronoi_nodes:
                available_lattices[voronoi_node.position] = voronoi_node.index

        # jump-flood until all lattices are filled
        lattices = np.array(tuple(facet_lattices), dtype=np.int32)
        step_size = 1
        while step_size < int((max(np.max(lattices[:, 0]) - np.min(lattices[:, 0]),
                                   np.max(lattices[:, 1]) - np.min(lattices[:, 1])) + 0) // 2 + 1):
            for voronoi_node in voronoi_nodes:
                # expand all territory in 8 directions
                for territory in [item for item in voronoi_node.territory]:
                    for key in [
                        tuple((territory[0] + step_size, territory[1])),
                        tuple((territory[0] - step_size, territory[1])),
                        tuple((territory[0], territory[1] + step_size)),
                        tuple((territory[0], territory[1] - step_size)),
                        tuple((territory[0] + step_size, territory[1] + step_size)),
                        tuple((territory[0] + step_size, territory[1] - step_size)),
                        tuple((territory[0] - step_size, territory[1] + step_size)),
                        tuple((territory[0] - step_size, territory[1] - step_size)),
                    ]:
                        if key in available_lattices.keys():
                            if available_lattices[key] == -1:
                                available_lattices[key] = voronoi_node.index
                                voronoi_node.territory.add(key)
                                voronoi_node.count += 1
                            elif available_lattices[key] == voronoi_node.index:
                                pass
                            elif np.linalg.norm(voronoi_nodes[available_lattices[key]].numpy - np.array(key,
                                                                                                        dtype=np.int32)) < np.linalg.norm(
                                    voronoi_node.numpy - np.array(key, dtype=np.int32)):
                                pass
                            else:
                                voronoi_nodes[available_lattices[key]].territory.remove(key)
                                voronoi_nodes[available_lattices[key]].count -= 1
                                available_lattices[key] = voronoi_node.index
                                voronoi_node.territory.add(key)
                                voronoi_node.count += 1
                        # increase accuracy, but slow
                        # else:
                        #     voronoi_node.territory.add(key)

            step_size = step_size * 2

        # what else?
        print("painting")
        color_bar = {i: tuple(float(item) for item in [abs(np.sin(10 * i)), abs(np.cos(i)), abs(np.sin(2 * i)), ]) for i
                     in range(10000)}
        color_bar[-1] = tuple([1.0, 0.0, 0.0])
        import matplotlib.pyplot as plt
        for point in available_lattices.keys():
            plt.scatter(0.3333 + point[0] * self.radius * self.pixel_factor,
                        0.3333 + point[1] * self.radius * self.pixel_factor,
                        color=tuple(float(i) for i in color_bar[available_lattices[point]]), s=10)
        for point in voronoi_nodes:
            plt.scatter(0.3333 + point.position[0] * self.radius * self.pixel_factor,
                        0.3333 + point.position[1] * self.radius * self.pixel_factor, color=(0.0, 0.0, 0.0), s=10)

        plt.show()

        return voronoi_nodes, available_lattices

    def relaxation(self, voronoi_nodes):
        """
        Lloyd
        :return:
        """
        for voronoi_node in voronoi_nodes:
            center = np.sum(np.array([pos for pos in voronoi_node.territory], dtype=np.float32),
                            axis=0) / voronoi_node.count
            direction = center - voronoi_node.numpy
            voronoi_node.move(
                tuple([round(voronoi_node.numpy[0] + direction[0]), round(voronoi_node.numpy[1] + direction[1])]))
        return voronoi_nodes

    def dynamic_relaxation(self, voronoi_nodes, available_lattices=None):
        for voronoi_node in voronoi_nodes:
            center = np.sum(np.array([pos for pos in voronoi_node.territory], dtype=np.float32),
                            axis=0) / voronoi_node.count
            direction = center - voronoi_node.numpy
            # if voronoi_node.count > ...:
            #     voronoi_node.move(tuple([round(voronoi_node.numpy[0]+direction[0]), round(voronoi_node.numpy[1]+direction[1])]))
            # else:
            neighborhood = set()
            for i in range(round(2 // self.pixel_factor)):
                for j in range(round(2 // self.pixel_factor)):
                    neighborhood.add(tuple((voronoi_node.position[0] + i, voronoi_node.position[1] + j)))
                    neighborhood.add(tuple((voronoi_node.position[0] + i, voronoi_node.position[1] - j)))
                    neighborhood.add(tuple((voronoi_node.position[0] - i, voronoi_node.position[1] + j)))
                    neighborhood.add(tuple((voronoi_node.position[0] - i, voronoi_node.position[1] - j)))
            resistance = np.zeros_like(direction)
            for node in voronoi_nodes:
                if node.index != voronoi_node.index:
                    if node.position in neighborhood:
                        difference = (voronoi_node.numpy - node.numpy)
                        if np.linalg.norm(difference) < round(2 // self.pixel_factor):
                            resistance += difference * (1 - np.linalg.norm(difference) / round(2 // self.pixel_factor))
            destination = tuple([round(voronoi_node.numpy[0] + direction[0] + resistance[0]),
                                 round(voronoi_node.numpy[1] + direction[1] + resistance[1])])
            if destination in available_lattices:
                voronoi_node.move(destination)
            else:
                voronoi_node.move(
                    tuple([round(voronoi_node.numpy[0] + direction[0]), round(voronoi_node.numpy[1] + direction[1])]))

        return voronoi_nodes

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
    radius = 0.05
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    facet = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    half_edge_mesh = HalfEdgeMesh(vertices, facet)
    voronoi = Voronoi(radius)
    voronoi_nodes, available_lattices = voronoi.generation(list(half_edge_mesh.half_edge_facets.values())[0])
    print("done")
    for i in range(50):
        voronoi_nodes = voronoi.dynamic_relaxation(voronoi_nodes, available_lattices)
        voronoi_nodes, available_lattices = voronoi.generation(list(half_edge_mesh.half_edge_facets.values())[0],
                                                               seeds=voronoi_nodes)
        print("done")
    # color_bar = {i: tuple(float(item) for item in [abs(np.sin(10*i)), abs(np.cos(i)), abs(np.sin(2 * i)), ]) for i in
    #              range(10000)}
    # color_bar[-1] = tuple([1.0, 0.0, 0.0])
    # import matplotlib.pyplot as plt
    # for point in available_lattices.keys():
    #     plt.scatter(0.3333+point[0]*radius*0.5, 0.3333+point[1]*radius*0.5, color=tuple(float(i) for i in color_bar[available_lattices[point]]))
    # for point in voronoi_nodes:
#
#     plt.scatter(0.3333+point.position[0]*radius*0.5, 0.3333+point.position[1]*radius*0.5, color=(0.0, 0.0, 0.0))
#
# plt.show()
