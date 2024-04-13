import numpy as np
from utils.half_edge_mesh import HalfEdgeMesh, HalfEdge, HalfEdgeVertex, HalfEdgeFacet
from collections import deque


class QuickConvexHull:
    def __init__(self, point_cloud):
        self.point_cloud = HalfEdgeMesh(point_cloud, [])
        self.initial_tetrahedron = self.make_tetrahedron()
        self.deque = deque()
        potential_outside_vertices = set(self.point_cloud.half_edge_vertices.values())
        for facet in self.initial_tetrahedron.half_edge_facets.values():
            facet, outside_vertices, _ = self.get_outside_vertices(facet, potential_outside_vertices)
            self.deque.append([facet, outside_vertices])
            potential_outside_vertices.difference_update(outside_vertices)
        self.iteration()

    def make_tetrahedron(self):
        # find 6 point with the largest x, y, z
        max_x = [-np.inf, -1]
        max_y = [-np.inf, -1]
        max_z = [-np.inf, -1]
        min_x = [np.inf, -1]
        min_y = [np.inf, -1]
        min_z = [np.inf, -1]
        for vertex in self.point_cloud.half_edge_vertices.values():
            if vertex.x > max_x[0]:
                max_x = vertex.x, vertex
            if vertex.y > max_y[0]:
                max_y = vertex.y, vertex
            if vertex.z > max_z[0]:
                max_z = vertex.z, vertex
            if vertex.x < min_x[0]:
                min_x = vertex.x, vertex
            if vertex.y < min_y[0]:
                min_y = vertex.y, vertex
            if vertex.z < min_z[0]:
                min_z = vertex.z, vertex
        base_points = [max_x[1], max_y[1], max_z[1], min_x[1], min_y[1], min_z[1]]
        # build first line with the most distant 2 points
        _, vertex1, vertex2 = sorted(
            [(np.linalg.norm(p1 - p2), p1, p2) for _, p1 in enumerate(base_points[:-1]) for p2 in base_points[_ + 1:]])[
            -1]
        half_edge1 = HalfEdge(vertex1, None, None, None, 0)
        half_edge1p = HalfEdge(vertex2, None, None, None, 3)
        half_edge1.pair = half_edge1p
        half_edge1p.pair = half_edge1
        # build first plane with the most distant point to the first line
        max_distance = 0.0
        vertex3 = None
        for vertex in self.point_cloud.half_edge_vertices.values():
            distance = self.get_distance(vertex, half_edge1)
            if distance > max_distance:
                max_distance = distance
                vertex3 = vertex
        half_edge2 = HalfEdge(vertex2, None, None, None, 1)
        half_edge2p = HalfEdge(vertex3, None, None, None, 4)
        half_edge2.pair = half_edge2p
        half_edge2p.pair = half_edge2
        half_edge3 = HalfEdge(vertex3, None, None, None, 2)
        half_edge3p = HalfEdge(vertex1, None, None, None, 5)
        half_edge3.pair = half_edge3p
        half_edge3p.pair = half_edge3
        half_edge1.next = half_edge2
        half_edge2.next = half_edge3
        half_edge3.next = half_edge1
        facet1 = HalfEdgeFacet(0)
        facet1.half_edge[half_edge1.index] = half_edge1
        facet1.half_edge[half_edge2.index] = half_edge2
        facet1.half_edge[half_edge3.index] = half_edge3
        # find forth vertex most distant to the first plane
        max_distance = 0.0
        vertex4 = None
        for vertex in self.point_cloud.half_edge_vertices.values():
            distance = self.get_distance(vertex, facet1)
            if abs(distance) > abs(max_distance):
                max_distance = distance
                vertex4 = vertex
        vertices = [vertex1.numpy, vertex2.numpy, vertex3.numpy, vertex4.numpy]
        if max_distance > 0:
            # flip facet1
            facets = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
        else:
            facets = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        return HalfEdgeMesh(vertices, facets)

    def iteration(self, iterations=0):
        count = 0
        while self.deque:
            with open(r"C:\Users\cchen\PycharmProjects\djproject\utils\compute_flag.txt", 'r') as flag:
                if flag.read() == "stop":
                    return 0
                flag.close()
            count += 1
            print(len(self.deque))
            facet, outside_vertices = self.deque.pop()  # HalfEdgeFacet, [HalfEdgeVertex]
            if len(outside_vertices) > 0:  # need to upgrade
                # find the most distant vertex
                max_distance = 0.0
                farthest_vertex = None
                for vertex in outside_vertices:
                    distance = self.get_distance(vertex, facet)
                    if distance > max_distance:
                        max_distance = distance
                        farthest_vertex = vertex

                # check for all visible facets, initial facet is absolutely visible
                if max_distance > 0.0:
                    fv = HalfEdgeVertex(*farthest_vertex.numpy, "new_vertex")
                    visible_facets, new_facets = self.get_visible_facets(fv, facet)  # TODO
                else:
                    visible_facets, new_facets = {}, []
                # remove visible facets and corresponding half-edges, vertices
                # vertex_to_delete = {half_edge.vertex.index: half_edge.vertex for facet in visible_facets.values() for half_edge in facet.half_edge.values()}
                # for key in [half_edge.vertex.index for facet in new_facets for half_edge in facet.half_edge.values()]:
                #     if key in vertex_to_delete.keys():
                #         vertex_to_delete.pop(key)
                for facet in visible_facets.values():
                    self.initial_tetrahedron.delete_facet_and_cleanup_vertex(facet)
                # for vertex in vertex_to_delete.values():
                #     self.initial_tetrahedron.delete_vertex(vertex)
                # build new facets
                for facet in new_facets:
                    self.initial_tetrahedron.add_facet(facet)
                # for half_edge in self.initial_tetrahedron.half_edges.values():
                #     if half_edge.vertex == half_edge.next.vertex or half_edge.vertex == half_edge.next.next.vertex:
                #         print(half_edge)
                # update checklist
                potential_outside_vertices = set(
                    vertex_ for facet_, outside_vertices_ in self.deque for vertex_ in outside_vertices_ if
                    facet_ in visible_facets.values())
                potential_outside_vertices.update(outside_vertices)
                # remove facets visible in deque
                self.deque = deque([[facet_, outside_vertices_] for facet_, outside_vertices_ in self.deque if
                                    facet_ not in visible_facets.values()])
                # self.deque = deque([self.get_outside_vertices(facet, []) for facet in self.initial_tetrahedron.half_edge_facets.values()][::-1])
                # add new todos
                for facet in new_facets:
                    facet, outside_vertices, zero_distant_vertices = self.get_outside_vertices(facet, potential_outside_vertices)
                    if outside_vertices:
                        self.deque.appendleft([facet, outside_vertices])
                    potential_outside_vertices.difference_update(outside_vertices)
                    # potential_outside_vertices.difference_update(zero_distant_vertices)
                    # print(len(potential_outside_vertices), len(zero_distant_vertices))
            if count == iterations:
                break
            v_, f_ = self.initial_tetrahedron.export()
            self.initial_tetrahedron.save_obj(rf"utils\output\{count}.obj", v_, f_)

    def get_visible_facets(self, vertex: HalfEdgeVertex, facet: HalfEdgeFacet):
        visible_facets = dict()
        new_facets = []
        visible_facets[facet.index] = facet  # current facet is always visible
        facet_to_check = dict(facet.half_edge)  # 3 adjacent facets need to check
        while facet_to_check:  # check list is not empty
            _, half_edge = facet_to_check.popitem()  # pop one facet to check
            facet = half_edge.pair.facet
            if facet.index not in visible_facets.keys():  # this facet is not checked yet
                if self.get_distance(vertex, facet) > 0.0:  # this facet is visible
                    visible_facets[facet.index] = facet
                    facet_to_check.update(facet.half_edge)  # add adjacent facets to check list, ignore the checked ones
                # elif self.get_distance(vertex, facet) == 0.0:
                #     print(vertex, facet)
                else:  # this facet is not visible
                    edge = half_edge  # we need this edge to build a new facet
                    new_facet = HalfEdgeFacet(None)
                    new_half_edge1 = HalfEdge(edge.vertex, None, new_facet, None, None)
                    new_half_edge2 = HalfEdge(edge.next.vertex, None, new_facet, None, None)
                    new_half_edge3 = HalfEdge(vertex, None, new_facet, None, None)
                    new_half_edge1.next = new_half_edge2
                    new_half_edge2.next = new_half_edge3
                    new_half_edge3.next = new_half_edge1
                    new_facet.half_edge["new_half_edge1"] = new_half_edge1
                    new_facet.half_edge["new_half_edge2"] = new_half_edge2
                    new_facet.half_edge["new_half_edge3"] = new_half_edge3
                    new_facets.append(new_facet)  # new facet
            else:
                pass
        return visible_facets, new_facets

    def get_outside_vertices(self, facet, outside_vertices):
        check_list = set()
        zero_distant_vertices = set()
        if outside_vertices:  # search in these vertices
            for vertex in outside_vertices:
                distance = self.get_distance(vertex, facet)
                if distance > 0.0:
                    check_list.add(vertex)
                elif distance == 0.0:
                    zero_distant_vertices.add(vertex)
        else:  # search in all vertices
            for vertex in self.point_cloud.half_edge_vertices.values():
                if self.get_distance(vertex, facet) > 0.0:
                    check_list.add(vertex)
        return [facet, check_list, zero_distant_vertices]

    @staticmethod
    def get_distance(vertex, other):
        if type(other) is HalfEdgeVertex:
            return np.linalg.norm(vertex.numpy - other.numpy)
        elif type(other) is HalfEdge:
            vertex1 = vertex.numpy
            vertex2 = other.vertex.numpy
            vertex3 = other.pair.vertex.numpy
            area = np.linalg.norm(np.cross(vertex1 - vertex2, vertex1 - vertex3))
            if area == 0.0:
                return 0.0
            else:
                return area / np.linalg.norm(vertex2 - vertex3)
        elif type(other) is HalfEdgeFacet:
            normal = other.cal_normal()
            vertex_on_facet = next(iter(other.half_edge.values())).vertex.numpy
            distance = np.dot(vertex.numpy - vertex_on_facet, normal)
            if 0.0 <= distance <= 0.0001:
                return 0.0
            else:
                return distance


if __name__ == "__main__":
    vert, fac = HalfEdgeMesh.load_obj(r"test1.obj")
    ch = QuickConvexHull(vert)
    v, f = ch.initial_tetrahedron.export()
    ch.initial_tetrahedron.save_obj(rf"utils\output\final.obj", v, f)
