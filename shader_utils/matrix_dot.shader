#version 460 compatibility


layout(std430, binding=0) buffer MatrixA{
    dmat4 matrixA[];
};
layout(std430, binding=1) buffer MatrixB{
    dmat4 matrixB[];
};
layout(std430, binding=2) buffer MatrixC{
    dmat4 matrixC[];
};

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;

uniform uvec4 shapes;


void main() {
    uint i=gid/shapes.z;  // 0
    uint j=gid%shapes.z;  // 0
    for(uint x=0; x<shapes.y; ++x){
        matrixC[gid] += matrixB[x*shapes.z + j] * matrixA[i*shapes.y + x];
    }
    // matrixC[0] = matrixB[0]*matrixA[0] + matrixB[1]*matrixA[1];
}
