#version 460 compatibility

layout(std430, binding=0) buffer Particles{
    mat4x4 Particle[];
};
layout(std430, binding=1) buffer Facets{
    mat4x4 Facet[];
    // v1, v2, v3, index
    // e1, e2, e3, 0
    // nx, ny, nz, 0
    // 0 , 0 , 0 , 0
};
layout(std430, binding=2) buffer HalfEdges{
    mat4x4 HalfEdge[];
    // v1, v2, v3, index
    // e1, e2, e3, 0
    // nx, ny, nz, 0
    // 0 , 0 , 0 , 0
};


struct HalfEdgeVertice{
    vec3 pos;
    int index;
    float x;
    float y;
    float z;
    int half_edge[16];  // only 16 half-edges are allowed in this beta version
};

struct HalfEdge{
    int vertex;
    int facet;
    int pair;
    int next;
    int index;
};

struct HalfEdgeFacet{
    vec3 normal;
    float area;
    int index;
    int half_edge[3];  // only 3 half-edges are allowed in this beta version
};


void main() {

}
