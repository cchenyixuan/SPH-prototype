// not in use
#version 460 core

layout(location=0) in int v_index; // vertex id
out GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
}g_out;



layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 Particle[];
};
layout(std430, binding=1) buffer ParticlesSubData{
    // particle inside domain has additional data: t_transfer.xyz, 0.0, 0.0...;
    mat4x4 ParticleSubData[];
};
layout(std430, binding=2) buffer BoundaryParticles{
    // particle at boundary with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 BoundaryParticle[];
};
layout(std430, binding=5) coherent buffer VoxelParticleNumbers{
    int VoxelParticleNumber[];
};
layout(std430, binding=6) coherent buffer VoxelParticleInNumbers{
    int VoxelParticleInNumber[];
};
layout(std430, binding=7) coherent buffer VoxelParticleOutNumbers{
    int VoxelParticleOutNumber[];
};
layout(std430, binding=8) coherent buffer GlobalStatus{
    // simulation global settings and status such as max velocity etc.
    // [n_particle, n_boundary_particle, n_voxel, Inlet1ParticleNumber, Inlet2ParticleNumber, Inlet3ParticleNumber, Inlet1Pointer, Inlet2Pointer, Inlet3Pointer, Inlet1In, Inlet2In, Inlet3In]
    int StatusInt[];
};
layout(std430, binding=9) buffer GlobalStatus2{
    // simulation global settings and status such as max velocity etc.
    // [self.H, self.R, self.DELTA_T, self.VISCOSITY, self.COHESION, self.ADHESION, voxel_offset_x, voxel_offset_y, voxel_offset_z, Inlet1In_float, Inlet2In_float, Inlet3In_float]
    float StatusFloat[];
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
layout(std430, binding=15) coherent buffer Voxels5{
    int Voxel5[];
};
layout(std430, binding=16) buffer Inlets1{
    // inlet1 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; 0, 0, 0, rho; 0, 0, 0, P;
    mat4x4 Inlet1[];
};
layout(std430, binding=17) buffer Inlets2{
    // inlet2 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 Inlet2[];
};
layout(std430, binding=18) buffer Inlets3{
    // inlet3 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 Inlet3[];
};
const float PI = 3.141592653589793;
const int n_boundary_particle = 10383;
const int n_voxel = 48235;
const float h = 0.01;
const float r = 0.0025;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float rest_dense = 1000;
const float eos_constant = 32142.0;
const float delta_t = 0.00005;
const float viscosity = 0.001;
const float cohesion = 0.0001;
const float adhesion = 0.0001;
const vec3 offset = vec3(-0.434871, -0.690556, -0.245941);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 6.545e-08;

uniform int vector_type;
uniform mat4 projection;
uniform mat4 view;

void main() {
    g_out.v_pos = vec4(BoundaryParticle[v_index][0].xyz, 1.0); // set vertex position, w=1.0
    switch(vector_type){
        case 0:  // velocity
            g_out.v_color = vec4(0.0);
            break;
        case 1:  // acceleration
            g_out.v_color = vec4(0.0);
            break;
        // case2 show all boundary particle visibility
        case 2:
            g_out.v_color = vec4(BoundaryParticle[v_index][3].xyz/length(BoundaryParticle[v_index][3].xyz)*float(Status[7])/float(Status[8]), 1.0);
            break;
        // new shader same as above, named boundary vector vertex shader, change particle to boundaryparticle
        // rewrite g_out.v_color with pressure and cirection at BP[id][3].xyzw
    }
}
