import numpy as np


class Point:
    def __init__(self, x, y, z):
        self.position = np.array((x, y, z), dtype=np.float32)
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, point):
        return Point(*(self.position - point.position))

    def __add__(self, point):
        return Point(*(self.position + point.position))

    def __neg__(self):
        return Point(*-self.position)

    def __mul__(self, other):
        return Point(*(other*self.position))

    def __truediv__(self, other):
        return Point(*(self.position/other))

    def __eq__(self, point):
        return self.x == point.x and self.y == point.y and self.z == point.z

    def length(self):
        return np.linalg.norm(self.position)

    def __repr__(self):
        return f"<Point>: {self.x}, {self.y}, {self.z}"

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __getitem__(self, item):
        return self.position[item]


class Edge:
    def __init__(self, point_a, point_b):
        self.point_a = point_a
        self.point_b = point_b

    def __repr__(self):
        return f"<Edge>: {self.point_a}, {self.point_b}"

    def __hash__(self):
        return hash((self.point_a, self.point_b))

    def __eq__(self, edge):
        return (
                (self.point_a == edge.point_a) and (self.point_b == edge.point_b)) or \
            ((self.point_b == edge.point_a) and (self.point_a == edge.point_b)
             )

    def length(self):
        return np.linalg.norm(self.point_a-self.point_b)


class Facet:
    def __init__(self, point_a, point_b, point_c):
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        self.center = (self.point_a + self.point_b + self.point_c) / 3
        tmp = np.cross(self.point_b.position - self.point_a.position, self.point_c.position - self.point_b.position)
        self.normal = Point(*(tmp / np.linalg.norm(tmp)))

        self.edge_a = Edge(self.point_a, self.point_b)
        self.edge_b = Edge(self.point_b, self.point_c)
        self.edge_c = Edge(self.point_c, self.point_a)

    def flip_normal(self):
        self.point_b, self.point_c = self.point_c, self.point_b
        self.normal *= -1
        return self

    def get_distance(self, point):
        return np.dot(self.normal.position, point.position-self.center.position)

    def __eq__(self, other):
        if self.center == other.center:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.point_a, self.point_b, self.point_c))

    def __repr__(self):
        return f"<Facet>: Normal: {self.normal}, \n{self.edge_a}, \n{self.edge_b}, \n{self.edge_c}"

class Mesh:
    ...

class MeshIO:
    def __init__(self):
        self.file_name = None


class Block:
    def __init__(self, facet: Facet, point_cloud: np.ndarray):
        self.facet = facet
        self.point_cloud = point_cloud

    def find_farthest_point(self):
        facet_center = self.facet.center
        normal = self.facet.normal
        max_distance = 0.0
        farthest_point = None
        for vertex in self.point_cloud:
            v = vertex - facet_center
            # the true distance is thus calculated by: cos(\theta) \cdot distance = v \cdot n / |v|
            distance = np.dot(v, normal) / np.linalg.norm(v)
            if distance > max_distance:
                max_distance = distance
                farthest_point = vertex
        return farthest_point

    def build_new_block(self, check_list):
        farthest_point = self.find_farthest_point()
        new_blocks = []
        for block in check_list:
            # if we can see this facet at the vertex
            if np.dot(farthest_point - block.facet.center, block.facet.normal) >= 0:
                # we create 3 new block
                # the first block is farthest_point, block.facet.vertex_1, block.facet.vertex_2 and some vertices
                new_facet = Facet(farthest_point, block.facet.vertex_1, block.facet.vertex_2)
                particle_cloud = []
                remained_particle_cloud = []
                for particle in block.point_cloud:
                    if np.dot(particle - new_facet.center, new_facet.normal) > 0:
                        particle_cloud.append(particle)
                    else:
                        remained_particle_cloud.append(particle)
                new_blocks.append(Block(new_facet, np.array(particle_cloud)))
                # the second block is farthest_point, block.facet.vertex_2, block.facet.vertex_3 and some vertices
                new_facet = Facet(farthest_point, block.facet.vertex_2, block.facet.vertex_3)
                particle_cloud = []
                remained_remained_particle_cloud = []
                for particle in remained_particle_cloud:
                    if np.dot(particle - new_facet.center, new_facet.normal) > 0:
                        particle_cloud.append(particle)
                    else:
                        remained_remained_particle_cloud.append(particle)
                new_blocks.append(Block(new_facet, np.array(particle_cloud)))
                # the third block is farthest_point, block.facet.vertex_3, block.facet.vertex_1 and some vertices
                new_facet = Facet(farthest_point, block.facet.vertex_3, block.facet.vertex_1)
                particle_cloud = []
                for particle in remained_remained_particle_cloud:
                    if np.dot(particle - new_facet.center, new_facet.normal) > 0:
                        particle_cloud.append(particle)
                new_blocks.append(Block(new_facet, np.array(particle_cloud)))
        return new_blocks

    def __repr__(self):
        print(self.facet, self.point_cloud)
        return f"{self.facet}, {self.point_cloud}"


class QuickConvexHull:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.p1 = point_cloud[point_cloud[:, 0].argmax()]
        self.p2 = point_cloud[point_cloud[:, 0].argmin()]
        self.p3 = point_cloud[point_cloud[:, 1].argmax()]
        self.p4 = point_cloud[point_cloud[:, 1].argmin()]
        self.p5 = point_cloud[point_cloud[:, 2].argmax()]
        self.p6 = point_cloud[point_cloud[:, 2].argmin()]
        self.points = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6]
        sorted_p = sorted(
            [(np.linalg.norm(a - b), a, b) for step, a in enumerate(self.points[:-1]) for b in self.points[step + 1:]])
        p1 = sorted_p[0][1]
        p2 = sorted_p[0][2]

        self.check_list = []
        facet = Facet(self.p1, self.p5, self.p2)
        particle_cloud = []
        remained_particle_cloud = []
        for particle in self.point_cloud:
            if np.dot(particle - facet.center, facet.normal) > 0:
                particle_cloud.append(particle)
            else:
                remained_particle_cloud.append(particle)
        self.check_list.append(Block(facet, np.array(particle_cloud)))

        facet = Facet(self.p1, self.p3, self.p2)
        particle_cloud = []
        remained_remained_particle_cloud = []
        for particle in remained_particle_cloud:
            if np.dot(particle - facet.center, facet.normal) > 0:
                particle_cloud.append(particle)
            else:
                remained_remained_particle_cloud.append(particle)
        self.check_list.append(Block(facet, np.array(particle_cloud)))

        facet = Facet(self.p1, self.p2, self.p3)
        particle_cloud = []
        remained_remained_remained_particle_cloud = []
        for particle in remained_remained_particle_cloud:
            if np.dot(particle - facet.center, facet.normal) > 0:
                particle_cloud.append(particle)
            else:
                remained_remained_remained_particle_cloud.append(particle)
        self.check_list.append(Block(facet, np.array(particle_cloud)))

        facet = Facet(self.p2, self.p5, self.p3)
        particle_cloud = []
        for particle in remained_remained_remained_particle_cloud:
            if np.dot(particle - facet.center, facet.normal) > 0:
                particle_cloud.append(particle)
        self.check_list.append(Block(facet, np.array(particle_cloud)))
        self.ans = []

    def recursive_quick_hull(self):
        if not self.check_list:
            return
        print([])
        block = self.check_list.pop()
        if block.point_cloud.shape[0] == 0:
            self.ans.append(block)
            return self.recursive_quick_hull()
        self.check_list = block.build_new_block([block, *self.check_list])
        return self.recursive_quick_hull()


if __name__ == "__main__":
    data = np.load("convexhulltest.npy")
    ch = QuickConvexHull(data)
    ch.recursive_quick_hull()
