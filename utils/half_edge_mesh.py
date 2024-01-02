import numpy as np


class HalfEdge:
    def __init__(self, vertex, pair_half_edge, facet, next_half_edge, index):
        self.vertex = vertex
        self.pair = pair_half_edge
        self.facet = facet
        self.next = next_half_edge
        self.index = index

    def __hash__(self):
        return self.index


class HalfEdgeVertex:
    def __init__(self, x, y, z, index):
        self.x = x
        self.y = y
        self.z = z
        self.numpy = np.array((self.x, self.y, self.z), dtype=np.float32)
        self.half_edge = set()
        self.index = index

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __sub__(self, other):
        return self.numpy - other.numpy


class HalfEdgeFacet:
    def __init__(self, index):
        self.half_edge = set()
        self.index = index

    def nearby(self, adjoins="edge"):
        if adjoins == "edge":
            return [half_edge.pair.facet for half_edge in self.half_edge]
        elif adjoins == "vertex":
            return [half_edge.pair.facet for half_edge in self.half_edge]

    def normal(self):
        vertex = []
        for half_edge in self.half_edge:
            vertex.append(half_edge.vertex)
        tmp = np.cross(vertex[1] - vertex[0], vertex[2] - vertex[1])
        return tmp / np.linalg.norm(tmp)


class HalfEdgeMesh:
    def __init__(self, vertices, facets):
        # facet -> 3 vertex -> 6 half-edge
        self.half_edges = {}
        self.half_edge_vertices = {}
        self.half_edge_facets = {}
        # arrange all vertices
        for step, vertex in enumerate(vertices):
            self.half_edge_vertices[step] = HalfEdgeVertex(*vertex, step)
        # build up geometry
        # for each half-edge, set its vertex, pair, facet, next, index
        # for each vertex, set its xyz, half-edge, index
        # for each facet, set its half-edge, index
        for step, facet in enumerate(facets):
            self.half_edge_facets[step] = HalfEdgeFacet(step)  # facet index
            # make 3 half-edges, pair and next undefined
            self.half_edges[3 * step + 0] = HalfEdge(self.half_edge_vertices[facet[0]], None, self.half_edge_facets[step],
                                                     None, 3 * step + 0)
            self.half_edges[3 * step + 1] = HalfEdge(self.half_edge_vertices[facet[1]], None, self.half_edge_facets[step],
                                                     None, 3 * step + 1)
            self.half_edges[3 * step + 2] = HalfEdge(self.half_edge_vertices[facet[2]], None, self.half_edge_facets[step],
                                                     None, 3 * step + 2)
            # set next for these edges
            self.half_edges[3 * step + 0].next = self.half_edges[3 * step + 1]
            self.half_edges[3 * step + 1].next = self.half_edges[3 * step + 2]
            self.half_edges[3 * step + 2].next = self.half_edges[3 * step + 0]
            # set pair for these edges
            for half_edge in self.half_edge_vertices[facet[1]].half_edge:  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[0]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 0].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 0]  # set pair to each other
            for half_edge in self.half_edge_vertices[facet[2]].half_edge:  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[1]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 1].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 1]  # set pair to each other
            for half_edge in self.half_edge_vertices[facet[0]].half_edge:  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[2]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 2].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 2]  # set pair to each other
            # set facet half-edge
            self.half_edge_facets[step].half_edge.add(self.half_edges[3 * step + 0])
            self.half_edge_facets[step].half_edge.add(self.half_edges[3 * step + 1])
            self.half_edge_facets[step].half_edge.add(self.half_edges[3 * step + 2])
            # set vertex half-edge
            self.half_edge_vertices[facet[0]].half_edge.add(self.half_edges[3 * step + 0])
            self.half_edge_vertices[facet[1]].half_edge.add(self.half_edges[3 * step + 1])
            self.half_edge_vertices[facet[2]].half_edge.add(self.half_edges[3 * step + 2])

    def export(self):
        vertices = np.array([vertex.numpy for vertex in self.half_edge_vertices.values()])
        facets = np.array(
            [[half_edge.vertex.index + 1 for half_edge in facet.half_edge] for facet in self.half_edge_facets.values()])
        return vertices, facets


def load_file(file):
    vertices = []
    faces = []
    try:
        with open(file, "r") as f:
            for row in f:
                if row[:2] == "v ":
                    vertices.append(row[2:-1].split(" "))
                elif row[:2] == "f ":
                    faces.append([item.split("/")[0] for item in row[2:-1].split(" ")])
            f.close()
    except FileNotFoundError:
        pass
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32) - np.array([1, 1, 1])


if __name__ == "__main__":
    geometry = r"D:\ProgramFiles\PycharmProject\SPH-prototype\utils\convexhull_test.obj"
    v, fac = load_file(geometry)
    hf = HalfEdgeMesh(v, fac)
    v, fac = hf.export()
    with open(r"D:\ProgramFiles\PycharmProject\SPH-prototype\models\convexhull_test2.obj", "w") as f_:
        f_.write("o text.obj\n")
        for vertex_ in v:
            f_.write(f"v {vertex_[0]} {vertex_[1]} {vertex_[2]}\n")
        for facet_ in fac:
            f_.write(f"f {facet_[0]} {facet_[1]} {facet_[2]}\n")
        f_.close()
