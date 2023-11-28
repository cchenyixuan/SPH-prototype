#version 460 compatibility

struct Matrix{
    mat4 a[4];
    mat4 b[4];
    mat4 c[4];
    mat4 d[4];
};

layout(std430, binding=0) buffer MatrixA{
    mat4 matrixA[];
};
layout(std430, binding=1) buffer MatrixB{
    mat4 matrixB[];
};
layout(std430, binding=2) buffer MatrixC{
    mat4 matrixC[];
};

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;

uniform uvec4 shapes;


void main() {
    uint ai = gid/(shapes.y*shapes.z);
    uint aj = gid%shapes.y;
    uint bi = gid%shapes.y;
    uint bj = (gid/shapes.y)%shapes.z;
    uint ci = gid/(shapes.y*shapes.z);  // == ai
    uint cj = (gid/shapes.y)%shapes.z;  // == bj
    matrixC[ci*shapes.z + cj] += matrixA[ai*shapes.y + aj] * matrixB[bi*shapes.z +bj];
}
