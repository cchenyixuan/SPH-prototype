import numpy as np
import random


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
        self.center = sum(self.vertices_buffer)/self.length
        self.normal = np.zeros_like(self.vertices_buffer[0])
        for node in self.start:
            v1 = node.position
            v2 = node.front.position
            v0 = self.center
            cross = np.cross(v1-v0, v2-v0)
            cross /= np.linalg.norm(cross)
            self.normal += cross/self.length


class 






"""
1.  subdivide facets to make all rings have similar vertices number ~= max(ring1, ring2)
2.  the 2 rings have different mass center and average normal vector
3.  the interpolate path is defined by bezier curve of 4 points, the first 2 points are center of mass of the 2 rings, 
        others are defined by 2 parameters which transfers the center of mass along normal vector, moving length = paras

"""
if __name__ == "__main__":
    r = RingNode(np.array([1, 1, 1]))
    rr = Ring(np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32))