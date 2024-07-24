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

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;
int particle_index = int(gid)+1;
float particle_index_float = float(particle_index);


void main() {

}
