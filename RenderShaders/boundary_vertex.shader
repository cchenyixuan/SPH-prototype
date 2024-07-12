#version 460 core

// layout(location=0) in int v_index; // vertex id   TODO: disabled!!! replaced with gl_InstanceID
layout(location=1) in vec3 sphere_pos; // sphere vertex position
out vec4 v_color; // color output

layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    // x , y , z , voxel_id
    // vx, vy, vz, mass
    // wx, wy, wz, rho
    // ax, ay, az, pressure
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
layout(std430, binding=2) buffer BoundaryParticles{
    // particle at boundary with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    // x , y , z , voxel_id
    // vx, vy, vz, mass
    // wx, wy, wz, rho
    // ax, ay, az, pressure
    mat4x4 BoundaryParticle[];
};
layout(std430, binding=3) coherent buffer VoxelParticleNumbers{
    int VoxelParticleNumber[];
};
layout(std430, binding=4) coherent buffer VoxelParticleInNumbers{
    int VoxelParticleInNumber[];
};
layout(std430, binding=5) coherent buffer VoxelParticleOutNumbers{
    int VoxelParticleOutNumber[];
};
layout(std430, binding=6) buffer GlobalStatus{
    // simulation global settings and status such as max velocity etc.
    // [n_particle, n_boundary_particle, n_voxel, Inlet1ParticleNumber, Inlet2ParticleNumber, Inlet3ParticleNumber, Inlet1Pointer, Inlet2Pointer, Inlet3Pointer, Inlet1In, Inlet2In, Inlet3In]
    int StatusInt[];
};
layout(std430, binding=7) buffer Inlets1{
    // inlet1 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; 0, 0, 0, rho; 0, 0, 0, P;
    mat4x4 Inlet1[];
};
layout(std430, binding=8) buffer Inlets2{
    // inlet2 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 Inlet2[];
};
layout(std430, binding=9) buffer Inlets3{
    // inlet3 with n particles // particle inside domain with x, y, z, voxel_id; vx, vy, vz, mass; wx, wy, wz, rho; ax, ay, az, P;
    mat4x4 Inlet3[];
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


const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const int VOXEL_GROUP_SIZE = 300000;

int v_index = gl_InstanceID;

void main() {
    gl_Position = projection*view*vec4(BoundaryParticle[v_index][0].xyz + r*sphere_pos, 1.0); // set vertex position, w=1.0
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