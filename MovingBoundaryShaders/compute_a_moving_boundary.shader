#version 460 compatibility

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
layout(std430, binding=8) buffer GlobalStatus{
    // simulation global settings and status such as max velocity etc.
    // [n_particle, n_boundary_particle, n_voxel, voxel_memory_length, voxel_block_size, h_p, h_q, r_p, r_q, max_velocity_n-times_than_r, rest_dense, eos_constant, t_p, t_q, v_p, v_q, c_p, c_q, a_p, a_q]
    int StatusInt[];
};
layout(std430, binding=9) buffer GlobalStatus2{
    // simulation global settings and status such as max velocity etc.
    // [self.H, self.R, self.DELTA_T, self.VISCOSITY, self.COHESION, self.ADHESION]
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


layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint gid = gl_GlobalInvocationID.x;
int particle_index = int(gid)+139731+1;//73386+1;
float particle_index_float = float(particle_index);

const float PI = 3.141592653589793;

const int n_boundary_particle = StatusInt[0];
const int n_particle = StatusInt[1];
const int n_voxel = StatusInt[2];
const int voxel_memory_length = StatusInt[3];
const int voxel_block_size = StatusInt[4];
const int VOXEL_GROUP_SIZE = StatusInt[5];

const float h = StatusFloat[0];
const float r = StatusFloat[1];
const float delta_t = StatusFloat[2];
const float viscosity = StatusFloat[3];
const float cohesion = StatusFloat[4];
const float adhesion = StatusFloat[5];
const float rest_dense = StatusFloat[6];
const float eos_constant = StatusFloat[7];

uniform int current_step;

float current_time = delta_t*float(current_step);


const int pointer_a = 1000;  // pump A
const int pointer_b = 2000;  // pump B
const int pointer_c = 3000;  // gate C
const int pointer_d = 4000;  // gate D
const int pointer_e = 5000;  // fixed boundary E
const int pointer_f = 6000;  // moving boundary F

float GetCurrentMassPump(float time, int group_index){
    // time clipped in range 0-1, group_index refer to point-slice id
    float current_mass = 0.0;
    float id = float(group_index);
    float expected_time = time-id*0.0125;
    if(0.0 < expected_time && expected_time <= 0.1){current_mass = 0.005*(sin(-0.5*PI+20*PI*expected_time)+1.0)/2;}
    else{current_mass = 0.0;}
    return current_mass;
}

float GetCurrentMassGate(float time, int group_index){
    // time clipped in range 0-1, group_index refer to point-slice id
    float current_mass = 0.0;
    switch (group_index){
        case 0:
            if     (0.0 < time && time <= 0.1){current_mass = 0.001*(cos(10*PI*time)+1.0)/2;}
            else if(0.5 < time && time <= 0.6){current_mass = 0.001*(sin(-0.5*PI+10*PI*(time-0.5))+1.0)/2;}
            else if(0.1 < time && time <= 0.5){current_mass = 0.0;}
            else if(0.6 < time && time <= 1.0){current_mass = 0.001;}
            break;
        case 30:
            if     (0.0 < time && time <= 0.1){current_mass = 0.001*(sin(-0.5*PI+10*PI*time)+1.0)/2;}
            else if(0.5 < time && time <= 0.6){current_mass = 0.001*(cos(10*PI*(time-0.5))+1.0)/2;}
            else if(0.1 < time && time <= 0.5){current_mass = 0.001;}
            else if(0.6 < time && time <= 1.0){current_mass = 0.0;}
            break;
    }
    return current_mass;
}


void ComputeBoundaryMovement(){
    // our boundary changes their particle mass periodicly under control of global time
    // each period lasts 1.0s
    // special moving boundary particles saves their mass at BoundaryParticle[particle_index-1][1].w
    // type at BoundaryParticle[particle_index-1][2].x;
    // group_index at BoundaryParticle[particle_index-1][2].y;
    float clipped_time = current_time - float(int(current_time));
    if(int(round(BoundaryParticle[particle_index-1][2].x)) == 1){
        BoundaryParticle[particle_index-1][1].w = GetCurrentMassPump(clipped_time, int(round(BoundaryParticle[particle_index-1][2].y)));
    }
    // else if(int(round(BoundaryParticle[particle_index-1][2].x)) == 2){
    //     BoundaryParticle[particle_index-1][1].w = GetCurrentMassGate(clipped_time, int(round(BoundaryParticle[particle_index-1][2].y)));
    // }
}

void main() {
    ComputeBoundaryMovement();
}