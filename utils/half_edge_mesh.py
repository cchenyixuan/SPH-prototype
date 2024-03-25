import numpy as np
# TODO: keep facet orientation, facet.half_edge.remove, add keep same order


class HalfEdge:
    def __init__(self, vertex, pair_half_edge, facet, next_half_edge, index):
        self.vertex = vertex
        self.pair = pair_half_edge
        self.facet = facet
        self.next = next_half_edge
        self.index = index

    def __hash__(self):
        return hash(self.index)

    def __repr__(self):
        return f"HalfEdge Object at {hex(id(self))}, vertex={self.vertex}, index={self.index}."


class HalfEdgeVertex:
    def __init__(self, x, y, z, index):
        self.x = x
        self.y = y
        self.z = z
        self.numpy = np.array((self.x, self.y, self.z), dtype=np.float32)
        self.half_edge = set()
        self.index = index

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __sub__(self, other):
        return self.numpy - other.numpy

    def __lt__(self, other):
        return self.index < other.index

    def __repr__(self):
        return f"HalfEdgeVertex Object at {hex(id(self))}, vertex=({self.x}, {self.y}, {self.z}), index={self.index}."


class HalfEdgeFacet:
    def __init__(self, index):
        self.half_edge = set()
        self.index = index
        self.normal = None

    def __hash__(self):
        return hash(self.index)

    def __repr__(self):
        return f"HalfEdgeFacet Object at {hex(id(self))}, index={self.index}."

    def nearby(self, adjoins="edge"):
        if adjoins == "edge":
            return [half_edge.pair.facet for half_edge in self.half_edge]
        elif adjoins == "vertex":
            return [half_edge.pair.facet for half_edge in self.half_edge]

    def cal_normal(self):
        if self.normal is None:
            vertex = []
            half_edge = next(iter(self.half_edge))
            vertex.append(half_edge.vertex)
            vertex.append(half_edge.next.vertex)
            vertex.append(half_edge.next.next.vertex)
            tmp = np.cross(vertex[1] - vertex[0], vertex[2] - vertex[1])
            self.normal = tmp / np.linalg.norm(tmp)
        return self.normal


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
            self.half_edges[3 * step + 0] = HalfEdge(self.half_edge_vertices[facet[0]], None,
                                                     self.half_edge_facets[step],
                                                     None, 3 * step + 0)
            self.half_edges[3 * step + 1] = HalfEdge(self.half_edge_vertices[facet[1]], None,
                                                     self.half_edge_facets[step],
                                                     None, 3 * step + 1)
            self.half_edges[3 * step + 2] = HalfEdge(self.half_edge_vertices[facet[2]], None,
                                                     self.half_edge_facets[step],
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

        self.half_edge_vertices_rev = {self.half_edge_vertices[key]: key for key in self.half_edge_vertices.keys()}
        self.half_edges_rev = {self.half_edges[key]: key for key in self.half_edges.keys()}
        self.half_edge_facets_rev = {self.half_edge_facets[key]: key for key in self.half_edge_facets.keys()}

    # def delete_facet(self, facet):
    #     if facet in self.half_edge_facets_rev.keys():
    #         # facets
    #         facet = self.half_edge_facets.pop(self.half_edge_facets_rev.pop(facet))
    #         last_facet_index = max(self.half_edge_facets.keys())
    #         if facet.index == last_facet_index:  # last one
    #             pass
    #         else:
    #             shift_facet = self.half_edge_facets.pop(last_facet_index)
    #             self.half_edge_facets_rev.pop(shift_facet)
    #             shift_facet.index = facet.index
    #             self.half_edge_facets[facet.index] = shift_facet
    #             self.half_edge_facets_rev[shift_facet] = shift_facet.index
    #         # half-edges
    #         for half_edge in facet.half_edge:
    #             # print(half_edge.index, half_edge.vertex.index, *[item.index for item in half_edge.vertex.half_edge])
    #             if half_edge in half_edge.vertex.half_edge:  # remove from vertex
    #                 half_edge.vertex.half_edge.remove(half_edge)
    #             if half_edge in self.half_edges_rev.keys():  # remove from half-edge dict
    #                 self.delete_half_edge(half_edge)
    #             if half_edge.pair is not None:  # remove from pair
    #                 half_edge.pair.pair = None
    #             if len(half_edge.vertex.half_edge) == 0:  # remove vertex
    #                 self.delete_vertex(half_edge.vertex)

    # def delete_half_edge(self, half_edge):  # only used in self.delete_facet
    #     half_edge = self.half_edges.pop(self.half_edges_rev.pop(half_edge))
    #     last_half_edge_index = max(self.half_edges.keys())
    #     if half_edge.index == last_half_edge_index:  # last one
    #         pass
    #     else:
    #         shift_half_edge = self.half_edges.pop(last_half_edge_index)
    #         self.half_edges_rev.pop(shift_half_edge)
    #         shift_half_edge.vertex.half_edge.remove(shift_half_edge)
    #         shift_half_edge.index = half_edge.index
    #         shift_half_edge.vertex.half_edge.add(shift_half_edge)
    #         self.half_edges[half_edge.index] = shift_half_edge
    #         self.half_edges_rev[shift_half_edge] = shift_half_edge.index

    def delete_facet(self, facet):
        if facet not in self.half_edge_facets_rev:
            pass
        else:
            # remove a facet means remove 3 half-edges
            for half_edge in facet.half_edge:
                half_edge_index = half_edge.index
                if half_edge.pair is not None:
                    half_edge.pair.pair = None
                half_edge.vertex.half_edge.remove(half_edge)
                self.half_edges.pop(self.half_edges_rev.pop(half_edge))
                last_half_edge_index = max(self.half_edges.keys())
                if last_half_edge_index == half_edge_index:
                    pass
                else:
                    # switch the last half_edge to current index
                    shift_half_edge = self.half_edges.pop(last_half_edge_index)
                    self.half_edges_rev.pop(shift_half_edge)
                    shift_half_edge.vertex.half_edge.remove(shift_half_edge)
                    shift_half_edge.facet.half_edge.remove(shift_half_edge)
                    shift_half_edge.index = half_edge_index
                    if shift_half_edge.pair is not None:
                        shift_half_edge.pair.pair = shift_half_edge
                    shift_half_edge.next.next.next = shift_half_edge
                    shift_half_edge.vertex.half_edge.add(shift_half_edge)
                    shift_half_edge.facet.half_edge.add(shift_half_edge)
                    self.half_edges[half_edge_index] = shift_half_edge
                    self.half_edges_rev[shift_half_edge] = half_edge_index
            # remove facet
            facet_index = facet.index
            self.half_edge_facets.pop(self.half_edge_facets_rev.pop(facet))
            # shift the last facet to this index
            last_facet_index = max(self.half_edge_facets.keys())
            if last_facet_index == facet_index:
                pass
            else:
                shift_facet = self.half_edge_facets.pop(last_facet_index)
                self.half_edge_facets_rev.pop(shift_facet)
                shift_facet.index = facet_index
                self.half_edge_facets[facet_index] = shift_facet
                self.half_edge_facets_rev[shift_facet] = shift_facet.index

    def delete_edge(self, half_edge, half_edge_pair=None):
        if half_edge not in self.half_edges_rev:
            pass
        else:
            # remove an edge means remove 2 facets it connects and all references
            # remove facets
            self.delete_facet(half_edge.facet)
            if half_edge.pair is not None:
                self.delete_facet(half_edge.pair.facet)

    def delete_vertex(self, vertex):
        if vertex not in self.half_edge_vertices_rev:
            pass
        else:
            # remove itself, its edges, its edges' pair, its facets
            # facet, edge, edge-pair
            half_edge_to_delete = [*vertex.half_edge]
            for half_edge in half_edge_to_delete:
                self.delete_edge(half_edge, half_edge.pair)
            vertex_index = vertex.index
            self.half_edge_vertices.pop(self.half_edge_vertices_rev.pop(vertex))
            # shift the last vertex to its place
            last_vertex_index = max(self.half_edge_vertices.keys())
            if last_vertex_index == vertex_index:
                pass
            else:
                # switch the last vertex to current index
                shift_vertex = self.half_edge_vertices.pop(last_vertex_index)
                self.half_edge_vertices_rev.pop(shift_vertex)
                shift_vertex.index = vertex_index
                self.half_edge_vertices[vertex_index] = shift_vertex
                self.half_edge_vertices_rev[shift_vertex] = vertex_index
                # shift the last vertex to this index, change its refs
                # upgrade facets' half-edge and half-edge
                new_half_edge_set = set()
                for half_edge in shift_vertex.half_edge:
                    half_edge.facet.half_edge.remove(half_edge)
                    half_edge.vertex = shift_vertex
                    half_edge.facet.half_edge.add(half_edge)
                    new_half_edge_set.add(half_edge)
                shift_vertex.half_edge = new_half_edge_set

    def add_facet(self, vertices):
        # register vertex
        for vertex in vertices:
            if vertex in self.half_edge_facets_rev.keys():
                pass
            else:
                vertex.index = max(self.half_edge_vertices.keys())+1
                self.half_edge_vertices_rev[vertex] = vertex.index
                self.half_edge_vertices[vertex.index] = vertex
        # register half-edge
        half_edge1 = HalfEdge(vertices[0], None, None, None, None)
        half_edge2 = HalfEdge(vertices[1], None, None, None, None)
        half_edge3 = HalfEdge(vertices[2], None, None, None, None)
        # register facet
        ...

    def export(self):
        vertices = np.array([self.half_edge_vertices[key].numpy for key in sorted(self.half_edge_vertices.keys())])
        facets = np.array(
            [[half_edge.vertex.index + 1 for half_edge in facet.half_edge] for facet in self.half_edge_facets.values()])
        return vertices, facets

    @staticmethod
    def load_obj(file):
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

    @staticmethod
    def load_stl(file):
        vertices_dict = dict()
        vertices = set()
        faces = []
        try:
            with open(file, "r") as f:
                for row in f:
                    tmp = row[:-1].split(" ")
                    if len(tmp) >= 4 and tmp[-4] == "vertex":
                        x, y, z = (float(_) for _ in tmp[-3:])
                        vertex_index = len(vertices)
                        new_vertex = HalfEdgeVertex(x, y, z, vertex_index)
                        vertices.add(new_vertex)
                        if len(vertices) == vertex_index:  # this point already exists, find its index
                            faces.append(vertices_dict[(x, y, z)].index)
                        else:  # new vertex
                            vertices_dict[(x, y, z)] = new_vertex
                            faces.append(new_vertex.index)
                f.close()
        except FileNotFoundError:
            pass
        vertex_buffer = np.zeros((len(vertices), 3), dtype=np.float32)
        for vertex in vertices:
            vertex_buffer[vertex.index] = vertex.numpy
        return vertex_buffer, np.array(faces, dtype=np.int32).reshape((-1, 3))


if __name__ == "__main__":
    geometry = r"C:\Users\cchen\PycharmProjects\SPH-prototype-multi-version\models\vortex_object.obj"
    # geometry_stl = r"C:\Users\cchen\PycharmProjects\sph-prototype\m02_sample.stl"
    v, fac = HalfEdgeMesh.load_obj(geometry)
    hf = HalfEdgeMesh(v, fac)
    v, fac = hf.export()
    with open(r"test1.obj", "w") as f_:
        f_.write("o text.obj\n")
        for vertex_ in v:
            f_.write(f"v {vertex_[0]} {vertex_[1]} {vertex_[2]}\n")
        for facet_ in fac:
            f_.write(f"f {facet_[0]} {facet_[1]} {facet_[2]}\n")
        f_.close()

    for i in range(10):
        hf.delete_vertex(hf.half_edge_vertices[0])
        tmp = []
        for vertex in hf.half_edge_vertices_rev.keys():
            if len(vertex.half_edge) == 0:
                tmp.append(vertex)
        # for vertex in tmp:
        #     hf.delete_vertex(vertex)
    v, fac = hf.export()
    with open(r"test1.obj", "w") as f_:
        f_.write("o text.obj\n")
        for vertex_ in v:
            f_.write(f"v {vertex_[0]} {vertex_[1]} {vertex_[2]}\n")
        for facet_ in fac:
            f_.write(f"f {facet_[0]} {facet_[1]} {facet_[2]}\n")
        f_.close()

