#version 460 core

layout(location=0) in int v_index; // vertex id
out vec4 v_color; // color output

layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, 0, voxel_id; ux, uy, 0, 0; vx, vy, 0, 0; aux, auy, avx, avy;
    // x , y , 0 , voxel_id
    // grad(u).x, grad(u).y, grad(v).x, grad(v).y
    // lap(u), lap(v), u, v
    // du/dt, 0.0, 0.0, 0.0
    mat4x4 Particle[];
};
layout(std430, binding=1) buffer ParticlesSubData{
    // particle inside domain has additional data: t_transfer.xyz, 0.0, 0.0...;
    // 0 , 0 , 0 , 0
    // 0 , 0 , 0 , 0
    // 0 , 0 , 0 , 0
    // 0 , 0 , 0 , group_id
    mat4x4 ParticleSubData[];
};
layout(std430, binding=3) coherent buffer VoxelParticleNumbers{
    int VoxelParticleNumber[];
};
layout(std430, binding=6) coherent buffer GlobalStatus{
    // simulation global settings and status such as max velocity etc.
    // [n_particle, n_boundary_particle, n_voxel, Inlet1ParticleNumber, Inlet2ParticleNumber, Inlet3ParticleNumber, Inlet1Pointer, Inlet2Pointer, Inlet3Pointer, Inlet1In, Inlet2In, Inlet3In]
    int StatusInt[];
};
layout(std430, binding=10) coherent buffer Voxels0{
    // each voxel has 182 mat44 and first 2 matrices contains its id, x_offset of h, y_offset of h, z_offset of h; and neighborhood voxel ids
    // other 180 matrices containing current-indoor-particle-ids, particles getting out and particles stepping in
    // matrices are changed into integer arrays to apply atomic operations, first 32 integers for first 2 matrices and one voxel costs 2912 integers
    int Voxel0[];
};
layout(std430, binding=11) coherent buffer Voxels1{
    int Voxel1[];
};
layout(std430, binding=12) coherent buffer Voxels2{
    int Voxel2[];
};
layout(std430, binding=13) coherent buffer Voxels3{
    int Voxel3[];
};
layout(std430, binding=14) coherent buffer Voxels4{
    int Voxel4[];
};
layout(std430, binding=15) buffer GlobalStatus2{
    // simulation global settings and status such as max velocity etc.
    // [self.H, self.R, self.DELTA_T, self.VISCOSITY, self.COHESION, self.ADHESION, voxel_offset_x, voxel_offset_y, voxel_offset_z, Inlet1In_float, Inlet2In_float, Inlet3In_float]
    float StatusFloat[];
};


uniform mat4 projection;
uniform mat4 view;


const float PI = 3.141592653589793;
const int n_voxel = 646416;
const float h = 0.05;
const float r = 0.005;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float delta_t = 0.0000025;
const vec3 offset = vec3(-10.05, -0.05, -10.05);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 8.538886859432597e-05;


void main() {
    gl_Position = projection*view*vec4(BoundaryParticle[v_index][0].xyz, 1.0); // set vertex position, w=1.0
    v_color = vec4(0.5, 0.5, 0.5, 0.3);
    if (abs(BoundaryParticle[v_index][3].x) + abs(BoundaryParticle[v_index][3].y) + abs(BoundaryParticle[v_index][3].z) > 0.0000001){
        v_color = vec4(abs(BoundaryParticle[v_index][3].xyz)/length(BoundaryParticle[v_index][3].xyz), 0.3);
    }
    /*
    if     (BoundaryParticle[v_index][2].x == 1.0 && BoundaryParticle[v_index][1].w==0.0){
        v_color = vec4(0.0, 0.9, 0.0, 0.3);
    }
    else if(BoundaryParticle[v_index][2].x == 1.0 && BoundaryParticle[v_index][1].w>0.0){
        v_color = vec4(BoundaryParticle[v_index][1].w/0.005, 0.5, 0.5, 0.3);
    }
    */
    //v_color = vec4(abs(sin(float(voxel_id/2))), abs(cos(float(voxel_id/3))), abs(sin(float(voxel_id/5))), 0.3);
}