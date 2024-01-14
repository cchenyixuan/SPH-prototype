import random

import numpy as np


class HalfEdge:
    def __init__(self, vertex, pair_half_edge, facet, next_half_edge, index):
        self.vertex = vertex
        self.pair = pair_half_edge
        self.facet = facet
        self.next = next_half_edge
        self.index = index

    def __hash__(self):
        return hash(self.index)


class HalfEdgeVertex:
    def __init__(self, x, y, z, index):
        self.x = x
        self.y = y
        self.z = z
        self.numpy = np.array((self.x, self.y, self.z), dtype=np.float32)
        self.half_edge = dict()
        self.index = index

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __sub__(self, other):
        return self.numpy - other.numpy

    def __lt__(self, other):
        return self.index < other.index

    def __repr__(self):
        return f"{self.x}, {self.y}, {self.z}, {self.index}, {self.half_edge.keys()}"


class HalfEdgeFacet:
    def __init__(self, index):
        self.half_edge = dict()
        self.index = index
        self.normal = None

    def __hash__(self):
        return hash(self.index)

    def nearby(self, adjoins="edge"):
        if adjoins == "edge":
            return [half_edge.pair.facet for half_edge in self.half_edge.values()]
        elif adjoins == "vertex":
            return [half_edge.pair.facet for half_edge in self.half_edge.values()]

    def cal_normal(self):
        if self.normal is None:
            vertex = []
            half_edge = next(iter(self.half_edge.values()))
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
            for half_edge in self.half_edge_vertices[facet[1]].half_edge.values():  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[0]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 0].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 0]  # set pair to each other
            for half_edge in self.half_edge_vertices[facet[2]].half_edge.values():  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[1]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 1].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 1]  # set pair to each other
            for half_edge in self.half_edge_vertices[facet[0]].half_edge.values():  # all half-edge start from next vertex
                if self.half_edge_vertices[facet[2]] == half_edge.next.vertex:  # half-edge ends with same vertex
                    self.half_edges[3 * step + 2].pair = half_edge  # set pair to each other
                    half_edge.pair = self.half_edges[3 * step + 2]  # set pair to each other
            # set facet half-edge
            self.half_edge_facets[step].half_edge[self.half_edges[3 * step + 0].index] = self.half_edges[3 * step + 0]
            self.half_edge_facets[step].half_edge[self.half_edges[3 * step + 1].index] = self.half_edges[3 * step + 1]
            self.half_edge_facets[step].half_edge[self.half_edges[3 * step + 2].index] = self.half_edges[3 * step + 2]
            # set vertex half-edge
            self.half_edge_vertices[facet[0]].half_edge[self.half_edges[3 * step + 0].index] = self.half_edges[3 * step + 0]
            self.half_edge_vertices[facet[1]].half_edge[self.half_edges[3 * step + 1].index] = self.half_edges[3 * step + 1]
            self.half_edge_vertices[facet[2]].half_edge[self.half_edges[3 * step + 2].index] = self.half_edges[3 * step + 2]

        # self.half_edge_vertices_rev = {self.half_edge_vertices[key]: key for key in self.half_edge_vertices.keys()}
        # self.half_edges_rev = {self.half_edges[key]: key for key in self.half_edges.keys()}
        # self.half_edge_facets_rev = {self.half_edge_facets[key]: key for key in self.half_edge_facets.keys()}
        self.vertex_max_index = max(self.half_edge_vertices.keys()) if self.half_edge_vertices else -1
        self.half_edge_max_index = max(self.half_edges.keys()) if self.half_edges else -1
        self.facet_max_index = max(self.half_edge_facets.keys()) if self.half_edge_facets else -1

    def check_valid(self):
        if sorted(self.half_edge_facets.keys()) == sorted(range(self.facet_max_index+1)):
            print("Facet valid")
        if sorted(self.half_edges.keys()) == sorted(range(self.half_edge_max_index+1)):
            print("Edge valid")
        if sorted(self.half_edge_vertices.keys()) == sorted(range(self.vertex_max_index+1)):
            print("Vertex valid")

    def delete_facet(self, facet: HalfEdgeFacet):
        if facet.index not in self.half_edge_facets.keys():
            pass
        else:
            # remove a facet means remove 3 half-edges
            # TODO: the switch operation may be error due to the last could be inside the half_edge_to_delete
            switch_list = []
            for key in sorted(facet.half_edge.keys(), reverse=True):
                half_edge = facet.half_edge[key]
                if half_edge.pair is not None:
                    half_edge.pair.pair = None
                half_edge.vertex.half_edge.pop(half_edge.index)
                self.half_edges.pop(half_edge.index)
                if self.half_edge_max_index == half_edge.index:
                    self.half_edge_max_index -= 1
                    pass
                else:
                    # keep later to switch
                    switch_list.append(half_edge)
            # sort switch_list
            tmp = sorted([(item.index, step) for step, item in enumerate(switch_list)], reverse=True)
            tmp_ = []
            for item in tmp:
                tmp_.append(switch_list[item[1]])
            switch_list = tmp_
            for half_edge in switch_list:
                # switch the last half_edge to current index
                shift_half_edge = self.half_edges.pop(self.half_edge_max_index)
                self.half_edge_max_index -= 1

                shift_half_edge.vertex.half_edge.pop(shift_half_edge.index)
                shift_half_edge.facet.half_edge.pop(shift_half_edge.index)
                shift_half_edge.index = half_edge.index
                if shift_half_edge.pair is not None:
                    shift_half_edge.pair.pair = shift_half_edge
                shift_half_edge.next.next.next = shift_half_edge
                shift_half_edge.vertex.half_edge[shift_half_edge.index] = shift_half_edge
                shift_half_edge.facet.half_edge[shift_half_edge.index] = shift_half_edge

                self.half_edges[half_edge.index] = shift_half_edge
                # self.half_edges_rev[shift_half_edge] = half_edge_index

            # remove facet
            facet_index = facet.index
            self.half_edge_facets.pop(facet.index)
            # shift the last facet to this index
            last_facet_index = self.facet_max_index
            if last_facet_index == facet_index:
                self.facet_max_index -= 1
                pass
            else:
                shift_facet = self.half_edge_facets.pop(last_facet_index)
                # self.half_edge_facets_rev.pop(shift_facet)
                self.facet_max_index -= 1
                shift_facet.index = facet_index
                self.half_edge_facets[facet_index] = shift_facet
                # self.half_edge_facets_rev[shift_facet] = shift_facet.index

    def delete_facet_without_shift(self, facet: HalfEdgeFacet):
        if facet.index not in self.half_edge_facets.keys():
            pass
        else:
            # remove a facet means remove 3 half-edges
            # TODO: the switch operation may be error due to the last could be inside the half_edge_to_delete
            for key in sorted(facet.half_edge.keys(), reverse=True):
                half_edge = facet.half_edge[key]
                if half_edge.pair is not None:
                    half_edge.pair.pair = None
                half_edge.vertex.half_edge.pop(half_edge.index)
                self.half_edges.pop(half_edge.index)

            # remove facet
            facet_index = facet.index
            self.half_edge_facets.pop(facet.index)

    def delete_facet_and_cleanup_vertex(self, facet: HalfEdgeFacet):
        self.delete_facet(facet)
        for half_edge in facet.half_edge.values():
            if half_edge.vertex.half_edge:
                pass
            else:
                self.delete_vertex(half_edge.vertex)

    def delete_edge(self, half_edge: HalfEdge, half_edge_pair=None):
        if half_edge.index not in self.half_edges.keys():
            pass
        else:
            # remove an edge means remove 2 facets it connects and all references
            # remove facets
            self.delete_facet(half_edge.facet)
            if half_edge.pair is not None:
                self.delete_facet(half_edge.pair.facet)

    def delete_edge_and_cleanup_vertex(self, half_edge: HalfEdge, half_edge_pair=None):
        if half_edge.index not in self.half_edges.keys():
            pass
        else:
            # remove an edge means remove 2 facets it connects and all references
            # remove facets
            self.delete_facet_and_cleanup_vertex(half_edge.facet)
            if half_edge.pair is not None:
                self.delete_facet_and_cleanup_vertex(half_edge.pair.facet)

    def delete_vertex(self, vertex: HalfEdgeVertex):
        if vertex.index not in self.half_edge_vertices.keys():
            print(vertex)
            pass
        else:
            # remove itself, its edges, its edges' pair, its facets
            # facet, edge, edge-pair
            while vertex.half_edge.values():
                half_edge = next(iter(vertex.half_edge.values()))
                self.delete_edge(half_edge, half_edge.pair)
            vertex_index = vertex.index
            self.half_edge_vertices.pop(vertex.index)
            # shift the last vertex to its place
            last_vertex_index = self.vertex_max_index
            if last_vertex_index == vertex_index:
                self.vertex_max_index -= 1
                pass
            else:
                # switch the last vertex to current index
                shift_vertex = self.half_edge_vertices.pop(last_vertex_index)
                # self.half_edge_vertices_rev.pop(shift_vertex)
                self.vertex_max_index -= 1
                shift_vertex.index = vertex_index
                self.half_edge_vertices[vertex_index] = shift_vertex
                # self.half_edge_vertices_rev[shift_vertex] = vertex_index
                # shift the last vertex to this index, change its refs
                # upgrade facets' half-edge and half-edge
                for half_edge in shift_vertex.half_edge.values():
                    half_edge.facet.half_edge.pop(half_edge.index)
                    half_edge.vertex = shift_vertex
                    half_edge.facet.half_edge[half_edge.index] = half_edge

    def delete_vertex_without_shift(self, vertex: HalfEdgeVertex):
        if vertex.index not in self.half_edge_vertices.keys():
            print(vertex)
            pass
        else:
            # remove itself, its edges, its edges' pair, its facets
            # facet, edge, edge-pair
            while vertex.half_edge.values():
                half_edge = next(iter(vertex.half_edge.values()))
                self.delete_edge(half_edge, half_edge.pair)
            vertex_index = vertex.index
            self.half_edge_vertices.pop(vertex.index)

    def add_facet(self, facet: HalfEdgeFacet):
        for half_edge in facet.half_edge.values():
            self.add_half_edge(half_edge)
        if facet.index in self.half_edge_facets.keys():
            print("Facet Add Index Conflict!")
            pass
        else:
            index = self.facet_max_index + 1
            facet.index = index
            self.half_edge_facets[index] = facet
            self.facet_max_index += 1
            # arrange half-edge.pair
            for half_edge in facet.half_edge.values():
                for pair_half_edge in half_edge.next.vertex.half_edge.values():
                    if pair_half_edge.next.vertex == half_edge.vertex:
                        half_edge.pair = pair_half_edge
                        pair_half_edge.pair = half_edge

            facet_half_edge = dict()
            for half_edge in facet.half_edge.values():
                half_edge.vertex.half_edge[half_edge.index] = half_edge
                facet_half_edge[half_edge.index] = half_edge
            facet.half_edge = facet_half_edge
        return facet

    def add_half_edge(self, edge: HalfEdge):  # NOT CALLABLE INDIVIDUALLY
        self.add_vertex(edge.vertex)
        if edge.index in self.half_edges.keys():
            print("Half-Edge Add Index Conflict!")
            pass
        else:
            index = self.half_edge_max_index + 1
            edge.index = index
            self.half_edges[index] = edge
            self.half_edge_max_index += 1

    def add_edge(self):
        ...

    def add_vertex(self, vertex):
        if vertex.index in self.half_edge_vertices.keys():
            pass
        else:
            index = self.vertex_max_index+1
            vertex.index = index
            self.half_edge_vertices[index] = vertex
            self.vertex_max_index += 1

    def export(self):
        vertices = np.array([self.half_edge_vertices[key].numpy for key in sorted(self.half_edge_vertices.keys())])
        facets = [next(iter(facet.half_edge.values())) for facet in self.half_edge_facets.values()]
        facets = np.array([[item.vertex.index+1, item.next.vertex.index+1, item.next.next.vertex.index+1] for item in facets])
        return vertices, facets

    def export_obj(self, file):
        ...

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
    from random import choice
    random.seed(1)
    geometry = r"D:\ProgramFiles\PycharmProject\SPH-prototype\models\convexhull_test.obj"
    # geometry_stl = r"C:\Users\cchen\PycharmProjects\sph-prototype\m02_sample.stl"
    v, fac = HalfEdgeMesh.load_obj(geometry)
    hf = HalfEdgeMesh(v, fac)
    hf.check_valid()
    v, fac = hf.export()
    with open(r"test1.obj", "w") as f_:
        f_.write("o text.obj\n")
        for vertex_ in v:
            f_.write(f"v {vertex_[0]} {vertex_[1]} {vertex_[2]}\n")
        for facet_ in fac:
            f_.write(f"f {facet_[0]} {facet_[1]} {facet_[2]}\n")
        f_.close()

    ff = []
    for i in range(1000):
        ff.append(hf.half_edge_facets[0])
        hf.delete_facet(hf.half_edge_facets[0])

    hf.check_valid()
    v, fac = hf.export()
    with open(r"test1.obj", "w") as f_:
        f_.write("o text.obj\n")
        for vertex_ in v:
            f_.write(f"v {vertex_[0]} {vertex_[1]} {vertex_[2]}\n")
        for facet_ in fac:
            f_.write(f"f {facet_[0]} {facet_[1]} {facet_[2]}\n")
        f_.close()

