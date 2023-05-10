import numpy as np
from SpaceDivision import CreateVoxels


class Node:
    def __init__(self, index, label=0):
        self.index = index
        self.neighborhood_nodes = []
        self.lines = []
        self.triangles = []
        self.label = label

    def emit_lines(self):
        """
        only used to analyse mesh
        :return: list of Line objects
        """
        lines = []
        p0 = self.index
        for node in self.neighborhood_nodes:
            p1 = node.index
            if p0 < p1:
                lines.append(Line(None, p0, p1))
        return lines

    def emit_triangles(self):
        """
        only used to re-generate triangles
        :return: list of Triangle objects
        """
        if self.label == 0:
            triangles = []
            p0 = self.index
            while self.neighborhood_nodes:
                node = self.neighborhood_nodes.pop()
                p1 = node.index
                for sub_node in node.neighborhood_nodes:
                    for rest_node in self.neighborhood_nodes:
                        if sub_node.index == rest_node.index:
                            p2 = rest_node.index
                            triangles.append(Triangle(None, p0, p1, p2, None))
                # cut this connection
                for index, doppelganger in enumerate(node.neighborhood_nodes):
                    if self == doppelganger:
                        node.neighborhood_nodes.pop(index)
        else:  # label == 1
            triangles = []
            p0 = self.index
            while self.neighborhood_nodes:
                node = self.neighborhood_nodes.pop()
                p1 = node.index
                for sub_node in node.neighborhood_nodes:
                    for rest_node in self.neighborhood_nodes:
                        if sub_node.index == rest_node.index:
                            p2 = rest_node.index
                            # self, node, rest_node must on the original facets
                            for tri in self.triangles:
                                if tri in node.triangles and tri in rest_node.triangles:
                                    triangles.append(Triangle(None, p0, p1, p2, None))
                # cut this connection
                for index, doppelganger in enumerate(node.neighborhood_nodes):
                    if self == doppelganger:
                        node.neighborhood_nodes.pop(index)

        return triangles


class Line:
    def __init__(self, index, p0, p1):
        self.index = index
        vertices = sorted([p0, p1])
        self.vertex0 = vertices[0]
        self.vertex1 = vertices[1]

        self.nodes = []
        self.triangles = []

        self.new_node = None


class Triangle:
    def __init__(self, index, p0, p1, p2, normal):
        self.index = index
        vertices = sorted([p0, p1, p2])
        self.vertex0 = vertices[0]
        self.vertex1 = vertices[1]
        self.vertex2 = vertices[2]
        self.normal = np.array(normal, dtype=np.float32)

        self.nodes = []
        self.lines = []


class Mesh:
    def __init__(self, h, r, file=None):
        if file is None:
            self.vertices = np.array([[1, 0, 0, 1, 0, 0], [0, 3, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, -1, -1, -1]], dtype=np.float32)
            self.triangles = np.array([[0, 1, 2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)
        else:
            self.vertices, self.triangles = self.load_file(file)
        self.facets, self.nodes, self.lines = self.analyse_mesh()
        length_list = sorted([np.linalg.norm(line.vertex0-line.vertex1) for line in self.lines])
        self.min_length, self.max_length = length_list[0], length_list[-1]
        self.bounding_box = self.find_bounding_box()

    @staticmethod
    def load_file(file):
        import re
        find_vertex = re.compile(r"v (\+?-?[\d.]+) (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
        find_triangle = re.compile(r"f (\+?-?[\d.]+)/\d*/\d* (\+?-?[\d.]+)/\d*/\d* (\+?-?[\d.]+)/\d*/\d*\n", re.S)
        v_data = []
        f_data = []
        with open(file, "r") as f:
            for row in f:
                ans_v = re.findall(find_vertex, row)
                ans_f = re.findall(find_triangle, row)
                if ans_v:
                    ans_v = [float(ans_v[0][i]) for i in range(3)]
                    v_data.append([ans_v[0], ans_v[1], -ans_v[2]])
                if ans_f:
                    ans_f = [int(ans_f[0][i]) for i in range(3)]
                    f_data.append([ans_f[0]-1, ans_f[1]-1, ans_f[2]-1])
            f.close()
        return np.array(v_data, dtype=np.float32), np.array(f_data, dtype=np.int32)

    def ray_intersects_triangle(self, target_particle, ray, triangle):
        EPSILON = 1e-10
        vertex0 = self.vertices[triangle[0]][:3]
        vertex1 = self.vertices[triangle[1]][:3]
        vertex2 = self.vertices[triangle[2]][:3]

        edge1 = vertex1 - vertex0
        edge2 = vertex2 - vertex0
        h = np.cross(ray, edge2)
        a = np.dot(edge1, h)
        if -EPSILON < a < EPSILON:
            return False, None  # This ray is parallel to this triangle.
        f = 1.0 / a
        s = target_particle - vertex0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False, None
        q = np.cross(s, edge1)
        v = f * np.dot(ray, q)
        if v < 0.0 or u + v > 1.0:
            return False, None
        #  At this stage we can compute t to find out where the intersection point is on the line.
        t = f * np.dot(edge2, q)
        if t > EPSILON:  # ray intersection
            outIntersectionPoint = target_particle + ray * t
            return True, outIntersectionPoint
        else:
            # This means that there is a line intersection but not a ray intersection.
            return False, None

    def find_bounding_box(self):
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])
        min_z = np.min(self.vertices[:, 2])
        max_z = np.max(self.vertices[:, 2])
        return np.array((min_x, max_x, min_y, max_y, min_z, max_z), dtype=np.float32)

    def fill_bounding_box(self, r):
        min_x, max_x, min_y, max_y, min_z, max_z = self.bounding_box
        # Define the number of points along each axis
        num_points_x = round((max_x - min_x) / (2*r))
        num_points_y = round((max_y - min_y) / (2*r))
        num_points_z = round((max_z - min_z) / (2*r))

        # Generate evenly spaced coordinates along each axis
        x = np.linspace(min_x, min_x+2*r*(num_points_x-1), num_points_x)
        y = np.linspace(min_y, min_y+2*r*(num_points_y-1), num_points_y)
        z = np.linspace(min_z, min_z+2*r*(num_points_z-1), num_points_z)

        # Use meshgrid to create a grid of coordinates
        X, Y, Z = np.meshgrid(x, y, z)

        # Reshape the coordinate arrays into a (num_points**3, 3) array of points
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        return points

    def group_triangles(self, ray_centers, r, axis=0):
        # ray_centers: [[0, 0, 0], [0, 0, 1], ..., [0, 100, 100]], group by yz face
        group = {f"{center[1]},{center[2]}": [] for center in ray_centers}
        for tri in self.triangles:
            center = (self.vertices[tri[0]] + self.vertices[tri[1]] + self.vertices[tri[2]])/3
            y_id = int((center[1]+r-self.bounding_box[2]) / (2 * r))
            z_id = int((center[2]+r-self.bounding_box[4]) / (2 * r))
            group[f"{y_id},{z_id}"].append(tri)
        return group

    @staticmethod
    def create_facets(triangles):
        facets = [Triangle(step, *tri, None) for step, tri in enumerate(triangles)]
        return facets

    @staticmethod
    def create_lines(nodes):
        lines = []
        for node in nodes:
            lines.extend(node.emit_lines())
        for step, line in enumerate(lines):
            line.index = step
        return lines

    @staticmethod
    def create_nodes(vertices, facets):
        nodes = [Node(i) for i in range(vertices.shape[0])]
        for tri in facets:
            # upgrade neighborhood nodes
            nodes[tri.vertex0].neighborhood_nodes.extend(
                [nodes[tri.__getattribute__(f"vertex{i}")] for i in (1, 2) if nodes[tri.__getattribute__(f"vertex{i}")] not in nodes[tri.vertex0].neighborhood_nodes])
            nodes[tri.vertex1].neighborhood_nodes.extend(
                [nodes[tri.__getattribute__(f"vertex{i}")] for i in (0, 2) if nodes[tri.__getattribute__(f"vertex{i}")] not in nodes[tri.vertex1].neighborhood_nodes])
            nodes[tri.vertex2].neighborhood_nodes.extend(
                [nodes[tri.__getattribute__(f"vertex{i}")] for i in (1, 0) if nodes[tri.__getattribute__(f"vertex{i}")] not in nodes[tri.vertex2].neighborhood_nodes])
            # upgrade connected triangles
            nodes[tri.vertex0].triangles.append(tri)
            nodes[tri.vertex1].triangles.append(tri)
            nodes[tri.vertex2].triangles.append(tri)

        return nodes

    def analyse_mesh(self):
        facets = self.create_facets(self.triangles)
        nodes = self.create_nodes(self.vertices, facets)  # nodes: neighborhood nodes and triangles equipped
        lines = self.create_lines(nodes)
        for line in lines:
            nodes[line.vertex0].lines.append(line)
            nodes[line.vertex1].lines.append(line)  # nodes: lines equipped
            line.nodes.extend([nodes[line.vertex0], nodes[line.vertex1]])  # lines: nodes equipped
            # lines: triangles equipped
            for tri_a in nodes[line.vertex0].triangles:
                for tri_b in nodes[line.vertex1].triangles:
                    if tri_a == tri_b:
                        line.triangles.append(tri_a)
        # facets: nodes and lines equipped
        for tri in facets:
            tri.nodes = [nodes[tri.vertex0], nodes[tri.vertex1], nodes[tri.vertex2]]
            for line_a in nodes[tri.vertex0].lines:
                for line_b in nodes[tri.vertex1].lines:
                    if line_a == line_b:
                        tri.lines.append(line_a)
            for line_a in nodes[tri.vertex0].lines:
                for line_b in nodes[tri.vertex2].lines:
                    if line_a == line_b:
                        tri.lines.append(line_a)
            for line_a in nodes[tri.vertex1].lines:
                for line_b in nodes[tri.vertex2].lines:
                    if line_a == line_b:
                        tri.lines.append(line_a)

        return facets, nodes, lines

    def subdivide(self, iteration=1):
        for _ in range(iteration):
            # generate new nodes
            new_vertices = []
            new_nodes = []
            # p0 ---- pm ---- p1
            for step, line in enumerate(self.lines):
                p0 = self.vertices[line.vertex0]
                p1 = self.vertices[line.vertex1]
                pm = (p0 + p1) / 2
                new_vertices.append(pm)
                new_node = Node(len(self.nodes)+step, label=1)
                new_node.neighborhood_nodes = [*line.nodes]
                new_node.lines = [line]
                new_node.triangles = [*line.triangles]
                line.new_node = new_node
            # pa        pb
            #    \    /
            #      pm
            #    /    \
            # pc        pd
            for line in self.lines:
                if len(line.triangles) == 1:
                    t1 = line.triangles[0]
                    nearby_lines = []
                    for nearby_line in t1.lines:
                        if nearby_line != line:
                            nearby_lines.append(nearby_line)
                    la, lb = nearby_lines
                    line.new_node.neighborhood_nodes.extend([la.new_node, lb.new_node])
                    line.new_node.neighborhood_nodes = list(set(line.new_node.neighborhood_nodes))  # clear repeats
                    new_nodes.append(line.new_node)
                elif len(line.triangles) == 2:
                    t1 = line.triangles[0]
                    t2 = line.triangles[1]
                    nearby_lines = []
                    for nearby_line in t1.lines:
                        if nearby_line != line:
                            nearby_lines.append(nearby_line)
                    for nearby_line in t2.lines:
                        if nearby_line != line:
                            nearby_lines.append(nearby_line)
                    la, lb, lc, ld = nearby_lines
                    line.new_node.neighborhood_nodes.extend([la.new_node, lb.new_node, lc.new_node, ld.new_node])
                    line.new_node.neighborhood_nodes = list(set(line.new_node.neighborhood_nodes))  # clear repeats
                    new_nodes.append(line.new_node)
            # connect all feasible nodes
            for node in self.nodes:
                node.neighborhood_nodes = [line.new_node for line in node.lines]
            self.nodes.extend(new_nodes)
            self.vertices = np.vstack((self.vertices, np.array(new_vertices, dtype=np.float32)))
            triangles = [np.array((tri.vertex0, tri.vertex1, tri.vertex2), dtype=np.int32) for node in self.nodes for tri in node.emit_triangles()]
            self.triangles = np.vstack(triangles)

            self.facets, self.nodes, self.lines = self.analyse_mesh()
            self.min_length /= 2
            self.max_length /= 2

    def export_obj(self, export_dir="output.obj"):
        with open(export_dir, "w") as f:
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in self.triangles:
                f.write(f"f {t[0]+1}// {t[1]+1}// {t[2]+1}//\n")
            f.close()

    def export_boundary(self, r):
        while self.max_length > 2*r:
            self.subdivide()
        self.export_obj("boundary.obj")
        print("Boundary exported at 'boundary.obj'.")

    def export_internal(self, r):
        # while self.max_length > 2*r:
        #     self.subdivide()
        ray_centers = [(0, y, z) for y in range(int((self.bounding_box[3] - self.bounding_box[2]) // (2*r))+2) for z in range(int((self.bounding_box[5] - self.bounding_box[4]) / (2*r))+2)]
        # triangle_group = self.group_triangles(ray_centers, r)
        internel_points = []
        for point in self.fill_bounding_box(r):
            y_id = int(round((point[1]-self.bounding_box[2])/(2*r)))
            z_id = int(round((point[2] - self.bounding_box[4]) / (2 * r)))
            count = 0
            cross_point = []
            for tri in self.triangles:  # triangle_group[f"{y_id},{z_id}"]:
                ans = self.ray_intersects_triangle(point, 3*np.array((self.bounding_box[1]-self.bounding_box[0], 0, 0), dtype=np.float32), tri)
                if ans[0]:
                    apply = True
                    for cp in cross_point:
                        if np.linalg.norm(ans[1]-cp) < 0.00001:
                            apply = False
                    if apply:
                        count += 1
                        cross_point.append(ans[1])

            if count % 2 == 1:
                internel_points.append(point)
        return np.array(internel_points, dtype=np.float32)










class MeshToPointCloud:
    def __init__(self, mesh_file):
        self.mesh = mesh_file  # mesh.vertices: nx(x, y, z, nx, ny, nz), mesh.triangles: nx(p0, p1, p2)
        self.bounding_box = self.find_bounding_box(self.mesh)  # (min_x, max_x, min_y, max_y, min_z, max_z)
        self.diameter = self.bounding_box[1] - self.bounding_box[0]

    def __call__(self, radius=0.02, interval=0.0, flag="full"):
        if flag == "full":
            self.create_point_cloud(radius, interval)

    @staticmethod
    def find_bounding_box(mesh):
        min_x = np.min(mesh.vertices[:, 0])
        max_x = np.max(mesh.vertices[:, 0])
        min_y = np.min(mesh.vertices[:, 1])
        max_y = np.max(mesh.vertices[:, 1])
        min_z = np.min(mesh.vertices[:, 2])
        max_z = np.max(mesh.vertices[:, 2])
        return np.array((min_x, max_x, min_y, max_y, min_z, max_z), dtype=np.float32)

    def subdivide(self, min_length):
        new_triangles = []
        norm = np.linalg.norm
        for triangle in self.mesh.triangles:
            p0 = self.mesh.vertices[triangle[0]]
            p1 = self.mesh.vertices[triangle[1]]
            p2 = self.mesh.vertices[triangle[2]]
            if norm(p0[:3]-p1[:3]) > min_length or norm(p0[:3]-p2[:3]) > min_length or norm(p1[:3]-p2[:3]) > min_length:
                index = self.mesh.vertices.shape[0]
                pa = (p0 + p1) / 2  # index
                pb = (p1 + p2) / 2  # index+1
                pc = (p2 + p0) / 2  # index+2
                self.mesh.vertices = np.vstack((self.mesh.vertices, pa, pb, pc))
                t1 = np.array([triangle[0], index, index + 2], dtype=np.int32)
                t2 = np.array([index, triangle[1], index + 1], dtype=np.int32)
                t3 = np.array([index, index + 1, index + 2], dtype=np.int32)
                t4 = np.array([index + 2, index + 1, triangle[2]], dtype=np.int32)
                new_triangles.append(np.vstack((t1, t2, t3, t4)))
            else:  # remain unchanged
                new_triangles.append(triangle)
        self.mesh.triangles = np.vstack(new_triangles)
        # self.simplify()

    def subdivide_butterfly(self, num_iterations=1):
        vertices = self.mesh.vertices
        triangles = self.mesh.triangles

        for i in range(num_iterations):
            # Step 1: Compute the new vertices
            new_vertices = []
            for i, (v1, v2, v3) in enumerate(triangles):
                # Compute the average of the three vertices
                v = (vertices[v1] + vertices[v2] + vertices[v3]) / 3
                new_vertices.append(v)

            # Step 2: Create the new triangles
            new_triangles = []
            for i, (v1, v2, v3) in enumerate(triangles):
                # Compute the indices of the new vertices
                v4, v5, v6 = len(vertices) + 3 * i, len(vertices) + 3 * i + 1, len(vertices) + 3 * i + 2

                # Create the new triangles
                t1 = [v1, v4, v6]
                t2 = [v4, v2, v5]
                t3 = [v5, v3, v6]
                t4 = [v4, v5, v6]

                # Add the new triangles to the list
                new_triangles.extend([t1, t2, t3, t4])

            # Step 3: Connect the new triangles
            triangles = np.array(new_triangles, dtype=np.int32)
            vertices = np.vstack((vertices, np.array(new_vertices, dtype=np.float32)))
        self.mesh.vertices = vertices
        self.mesh.triangles = triangles

    def simplify(self):
        norm = np.linalg.norm
        last_ptr = self.mesh.vertices.shape[0]-1
        for i, pi in enumerate(self.mesh.vertices[:-1]):
            if i >= last_ptr:
                break
            for j, pj in enumerate(self.mesh.vertices[i+1:]):
                j += i+1
                if norm(pi-pj) < 1e-6:
                    # same position, merge
                    # switch with the last vertex and change index of triangles

                    self.mesh.vertices[j], self.mesh.vertices[last_ptr] = self.mesh.vertices[last_ptr], np.zeros_like(pi)
                    # for all triangles, change j to i; for all triangles, change last_ptr to j
                    for triangle in self.mesh.triangles:
                        if triangle[0] == j:
                            triangle[0] = i
                        elif triangle[1] == j:
                            triangle[1] = i
                        elif triangle[2] == j:
                            triangle[2] = i

                        if triangle[0] == last_ptr:
                            triangle[0] = j
                        elif triangle[1] == last_ptr:
                            triangle[1] = j
                        elif triangle[2] == last_ptr:
                            triangle[2] = j
                    # one vertex could only have one doppelganger, one found, others skip
                    last_ptr -= 1
                    break
        self.mesh.vertices = self.mesh.vertices[:last_ptr+1]


    def create_point_cloud(self, radius, interval):
        ...





if __name__ == "__main__":
    mesh = Mesh(0.02, 0.02/4, "fountain.obj")
    internal = mesh.export_internal(0.02/4)


