import sys

import numpy as np


def load_obj(file, uv=False, normal=False):
    print(f"Loading: {file}.")
    vertices = []
    faces = []
    uvs = []
    normals = []
    try:
        with open(file, "r", encoding="utf-8") as f:
            for row in f:
                if row[:2] == "v ":
                    vertices.append(row[2:-1].split(" "))
                elif row[:2] == "f ":
                    faces.append([item.split("/")[0] for item in row[2:-1].split(" ")])

                if uv:
                    if row[:3] == "vt ":
                        uvs.append(row[3:-1].split(" "))
                if normal:
                    if row[:3] == "vn ":
                        normals.append(row[3:-1].split(" "))
            f.close()
    except FileNotFoundError:
        print("File Not Found!", file=sys.stderr)
        pass
    print(f"Loading '{file}' finished with {len(vertices)} vertices and {len(faces)} facets.")
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32) - np.array([1, 1, 1])
