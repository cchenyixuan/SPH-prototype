import numpy as np
import random
import math


class RingNode:
    def __init__(self, position: np.ndarray, back=None, front=None, index=None):
        self.index = index
        self.position = position
        self.back = back
        self.front = front

    def __repr__(self):
        return f"Node {self.index}, Position {[*self.position]}, Index {self.index}, Id {id(self)}."

    def __iter__(self):
        current_node = self
        stop_index = current_node.index
        while True:
            yield current_node
            current_node = current_node.front
            if current_node.index == stop_index or current_node is None:
                break


class Ring:
    """
    vertices should be sorted as a ring
    """

    def __init__(self, vertices):
        self.start = RingNode(vertices[0], None, None, 0)
        self.length = 1
        last_node = self.start
        for step, vertex in enumerate(vertices[1:], start=1):
            new_node = RingNode(vertex, back=last_node, front=None, index=step)
            last_node.front = new_node
            last_node = last_node.front
            self.length += 1
        self.end = last_node
        self.end.front = self.start
        self.start.back = self.end
        print(f"Ring length: {self.length}")

        self.vertices_buffer = np.array([item.position for item in self.start], dtype=np.float32)
        self.center = sum(self.vertices_buffer) / self.length
        self.normal = np.zeros_like(self.vertices_buffer[0])
        for node in self.start:
            v1 = node.position
            v2 = node.front.position
            v0 = self.center
            cross = np.cross(v1 - v0, v2 - v0)
            cross /= np.linalg.norm(cross)
            self.normal += cross / self.length
        self.normal /= np.linalg.norm(self.normal)

    def __getitem__(self, item):
        item %= self.length
        for node in self.start:
            if node.index == item:
                return node
        print(f"No node match the index {item}!")

    def subdivide(self, times=1):
        if times == 0:
            # do nothing
            pass
        else:
            vertices = np.vstack([np.zeros_like(self.vertices_buffer) for i in range(times + 1)])
            for step, node in enumerate(self.start):
                vertices[step * (times + 1)] = node.position
                direction = (node.front.position - node.position) / (times + 1)
                for i in range(times):
                    vertices[step * (times + 1) + i + 1] = node.position + (i + 1) * direction
            self.__init__(vertices)

    def re_index(self, index):
        vertices = np.zeros_like(self.vertices_buffer)
        for step, node in enumerate(self.__getitem__(index)):
            vertices[step] = node.position
        self.__init__(vertices)

    def reverse(self):
        vertices = np.vstack([self.vertices_buffer[0], self.vertices_buffer[1:][::-1]])

        return Ring(vertices)


class BezierCurve:
    def __init__(self, points):
        self.degree = len(points) - 1
        self.function = lambda t: sum(
            [math.comb(self.degree, i) * points[i] * t ** i * (1 - t) ** (self.degree - i) for i in range(self.degree + 1)])
        self.length = lambda *args: sum([np.linalg.norm(self.function((t+1)*0.02)-self.function(t*0.02)) for t in range(50)])

    def __call__(self, *args):
        if args:
            return self.function(args[0])
        else:
            return self.function


class Connector:
    def __init__(self, ring1: Ring, ring2: Ring, interpolation=10, alpha=1.0, beta=1.0, index_offset=0):
        self.ring1 = ring1
        self.ring2 = ring2
        self.length = math.lcm(self.ring1.length, self.ring2.length)
        self.alpha = alpha
        self.beta = beta
        self.index_offset = index_offset
        # make each ring has same node number
        self.ring1.subdivide(self.length // self.ring1.length - 1)
        self.ring2.subdivide(self.length // self.ring2.length - 1)
        # align ring nodes
        self.align_ring_nodes()
        # get 2 rotation matrices describe the rotation status of r1 and r2 at half of the path
        v1 = self.ring1.normal
        v2 = self.ring2.normal
        # TODO: here is buggy: the curve seems to be reversed? i fix the output by reverse the curve but i cannot figure out the reason
        self.center_curve = BezierCurve(np.vstack([self.ring1.center, self.ring1.center + v1 * alpha, self.ring2.center + v2 * beta, self.ring2.center]))()
        # TODO: get the direction vector of the center_curve at t=0.5 ~= (curve(0.501)-curve(0.499)).normalize
        self.center_curve_middle_direction = -(self.center_curve(5.05)-self.center_curve(4.95))/np.linalg.norm(self.center_curve(5.05)-self.center_curve(4.95))
        self.ring1_half_rotation = self.get_rotation_matrix(np.sign(self.alpha)*self.ring1.normal, self.center_curve_middle_direction)
        self.ring2_half_rotation = self.get_rotation_matrix(-np.sign(self.beta)*self.ring2.normal, self.center_curve_middle_direction)
        # TODO: apply rotation

        self.curves = []
        self.p5s = []
        self.p1s = []
        self.p2s = []
        # the reverse depends on sign of alpha*beta (Hypothesis)
        for node1, node2 in zip(self.ring1.start, self.ring2.reverse().start) if alpha*beta > 0 else zip(self.ring1.start, self.ring2.start):
            p1 = node1.position
            p2 = node2.position
            p3 = node1.position + v1 * alpha
            p4 = node2.position + v2 * beta
            p5 = self.center_curve(0.5) + 0.5*(self.ring1_half_rotation@(p1-self.ring1.center)+self.ring2_half_rotation@(p2-self.ring2.center))
            self.p5s.append(self.center_curve(0.5) + self.ring1_half_rotation@(p1-self.ring1.center))
            self.p1s.append(p1)
            self.p2s.append(p2)
            self.curves.append(BezierCurve(np.vstack([p1, p3, p5, p5, p5, p4, p2]))())

    def __call__(self, *args, **kwargs):
        return self.curves

    def align_ring_nodes(self):
        # find the nearest points of the rings(with mass center merged and in opposite direction)
        # step1: calculate the rotation matrix
        rotation = self.get_rotation_matrix(np.sign(self.alpha)*self.ring1.normal, -np.sign(self.beta)*self.ring2.normal)
        # step2: align centers and rotate ring1 to get ring1_prime
        buffer1_prime = np.vstack(
            [rotation @ (node.position - self.ring1.center) + self.ring2.center for node in self.ring1.start])
        ring1_prime = Ring(buffer1_prime)
        # find the nearest 2 nodes from ring1_prime and ring2
        nearest_node_index = self.find_nearest_nodes(ring1_prime, self.ring2)
        # re-arrange
        self.ring1.re_index(nearest_node_index[0])
        self.ring2.re_index(nearest_node_index[1] + self.index_offset)
        # self.nearest_node_index = [0, 0]

    def get_rotation_matrix(self, vector1, vector2):
        if vector1@vector2 == -1:
            print("Opposite direction on the same axis, unable to define rotation matrix!")
            print("Randomly choose the rotation direction!")
            theta1, theta2, theta3 = np.random.random(3)/180  # 0-1degree
            sin_a, sin_b, sin_c, cos_a, cos_b, cos_c = [*np.sin([theta1, theta2, theta3]), *np.cos([theta1, theta2, theta3])]
            rotate_matrix = np.array([[cos_b*cos_c, sin_a*sin_b*cos_c-cos_a*sin_c, cos_a*sin_b*cos_c+sin_a*sin_c],
                                      [sin_c*cos_b, sin_a*sin_b*sin_c+cos_a*cos_c, cos_a*sin_b*sin_c-sin_a*cos_c],
                                      [-sin_b, sin_a*cos_b, cos_a*cos_b]], dtype=np.float32)
            return self.get_rotation_matrix(rotate_matrix@vector1, vector2)@rotate_matrix
        else:
            rotation = np.identity(3, dtype=np.float32)
            v = np.cross(vector1, vector2)
            c = np.dot(vector1, vector2)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
            rotation += vx + vx @ vx / (1 + c)
            return rotation

    def generate_guide_curve(self):
        ...

    def facet_construction(self):
        ...

    @staticmethod
    def find_nearest_nodes(ringA, ringB):
        nearest_node_index = [0, 0]
        min_distance = 1e8
        # naive solution O(mn)
        for node1 in ringA.start:
            for node2 in ringB.start:
                diff = node1.position - node2.position
                dis = diff @ diff
                if dis < min_distance:
                    min_distance = dis
                    nearest_node_index = [node1.index, node2.index]
        return nearest_node_index

    def bezier_path(self, a: float, b: float):
        """
        Bezier(p1, p2, v1, v2, a, b)
        4 points start at p1 end at p2, the interpolated 2 points are p3=p1+a*v1 and p4=p2+b*v2
        :param a: strength of v1
        :param b: strength of v2
        :return: a Bezier-curve function
        """
        return


class MeshIO:
    def __init__(self, file):
        self.vertex_data = self.load_file(file)

    @staticmethod
    def load_file(file):
        import re
        find_vertex = re.compile(r"v (\+?-?[\d.]+) (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
        find_line = re.compile(r"l (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
        v_data = []
        l_data = []
        with open(file, "r", encoding="utf-8") as f:
            for row in f:
                if row[:2] == "v ":
                    ans_v = re.findall(find_vertex, row)
                    ans_v = [float(ans_v[0][i]) for i in range(3)]
                    v_data.append([ans_v[0], ans_v[1], ans_v[2]])
                if row[:2] == "l ":
                    ans_l = re.findall(find_line, row)
                    ans_l = [int(ans_l[0][i]) for i in range(2)]
                    l_data.append([ans_l[0], ans_l[1]])
            f.close()
        # we can build a ring use the line data!
        # the data of line are connected as a ring, so we start from any line, connect the node to the next line
        start, current_point_index = l_data.pop()  # start from 1
        v_data_sorted = [v_data[start-1]]
        while l_data:
            for step, line in enumerate(l_data):
                if current_point_index in line:
                    start, current_point_index = l_data.pop(step)
                    v_data_sorted.append(v_data[start-1])
                    break
        return np.array(v_data_sorted, dtype=np.float32)


"""
1.  subdivide facets to make all rings have similar vertices number ~= max(ring1, ring2)
2.  the 2 rings have different mass center and average normal vector
3.  the interpolate path is defined by bezier curve of 4 points, the first 2 points are center of mass of the 2 rings, 
        others are defined by 2 parameters which transfers the center of mass along normal vector, moving length = paras

"""
# 1. sort the vertices read from raw mesh file
# TODO: 2. the two rings' vertices should be aligned (can be done by manually use offsets)
# TODO: 3. smoothed connection with the original geometry(reduced node number)
# TODO: 4. OpenGL visualization(GUI)
if __name__ == "__main__":
    r = RingNode(np.array([1, 1, 1]))

    o1 = MeshIO("ring1.obj").vertex_data
    o2 = MeshIO("ring2.obj").vertex_data
    r1 = Ring(o1)
    r2 = Ring(o2)
    c = Connector(r1, r2, 60, 4.2, 4.2, index_offset=0)
    interpolated_rings = [Ring(np.array([c.curves[j](i*1/60) for j in range(r1.length)], dtype=np.float32)) for i in range(61)]
    with open("a.obj", "w") as f:
        f.write("o opt\n")
        for ring in interpolated_rings:
            for node in ring.start:
                f.write(f"v {node.position[0]} {node.position[1]} {node.position[2]}\n")

        # for i in range(60):
        #     f.write(f"l {2929+i} {2930+i}\n")

        # for i in range(15):
        #     f.write(f"l {2990+i} {3038+i}\n")
        # f.write(f"l {1} {2}\n")
        # f.write(f"l {3} {4}\n")
        for i in range(len(interpolated_rings)-1):
            ring1, ring2 = interpolated_rings[i], interpolated_rings[i+1]
            for node1, node2 in zip(ring1.start, ring2.start):
                f.write(f"f {i*r1.length+node1.index+1} {(i+1)*r1.length+node2.index+1} {(i+1)*r1.length+node2.front.index+1} {i*r1.length+node1.front.index+1}\n")

        f.close()
