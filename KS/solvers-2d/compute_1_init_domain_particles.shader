#version 460 compatibility

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

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;
int particle_index = int(gid)+1;
float particle_index_float = float(particle_index);

const float PI = 3.141592653589793;
const int n_voxel = 244824;
const float h = 0.05;
const float r = 0.005;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float delta_t = 0.0000025;
const vec3 offset = vec3(-5.05, -0.05, -5.05);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 8.538886859432597e-05;


int get_voxel_data(int voxel_id, int pointer){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    int ans;
    switch(voxel_buffer_index){
        case 0:
            ans = Voxel0[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 1:
            ans = Voxel1[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 2:
            ans = Voxel2[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 3:
            ans = Voxel3[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 4:
            ans = Voxel4[voxel_local_index*voxel_memory_length+pointer];
            break;
        //case 5:
        //    ans = Voxel5[voxel_local_index*voxel_memory_length+pointer];
        //    break;
    }
    return ans;
}

void set_voxel_data(int voxel_id, int pointer, int value){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    value: value to set
    return: value before changed
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    switch(voxel_buffer_index){
        case 0:
            Voxel0[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 1:
            Voxel1[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 2:
            Voxel2[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 3:
            Voxel3[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 4:
            Voxel4[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        //case 5:
        //    Voxel5[voxel_local_index*voxel_memory_length+pointer] = value;
        //    break;
    }
}

int set_voxel_data_atomic(int voxel_id, int pointer, int value){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    value: value to set
    return: value before changed
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    int ans;
    switch(voxel_buffer_index){
        case 0:
            ans = atomicAdd(Voxel0[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 1:
            ans = atomicAdd(Voxel1[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 2:
            ans = atomicAdd(Voxel2[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 3:
            ans = atomicAdd(Voxel3[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 4:
            ans = atomicAdd(Voxel4[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        //case 5:
        //    ans = atomicAdd(Voxel5[voxel_local_index*voxel_memory_length+pointer], value);
        //    break;
    }
    return ans;
}

void AllocateParticles(){
    // position of current particle focused
    vec2 particle_pos = Particle[particle_index-1][0].xz;
    // for all voxels
    for(int i=0; i < n_voxel; ++i){
        // current voxel center position
        vec3 voxel_pos = offset + vec3(float(get_voxel_data(i+1, 1))*h, float(get_voxel_data(i+1, 2))*h, float(get_voxel_data(i+1, 3))*h);
        // current particle inside current voxel (vx-2/h<=px<vx+2/h)
        if(
            voxel_pos.x-h/2<=particle_pos.x && particle_pos.x<voxel_pos.x+h/2 &&
            voxel_pos.z-h/2<=particle_pos.y && particle_pos.y<voxel_pos.z+h/2
            ){
                // one particle found inside current voxel, get its slot id (start from 0) and add 1 to next slot id
                int c = atomicAdd(VoxelParticleNumber[i], 1);
                barrier();
                // set slot with index value
                set_voxel_data_atomic(i+1, 32+c%voxel_block_size, particle_index);  // starts from 1 (domain particle)
                barrier();
                // set particle's voxel id
                Particle[particle_index-1][0].w = float(i+1);  // starts from 1.0
                break;
        };

    }
    //if(Particle[particle_index-1][0].w<0.5){
    //    Particle[particle_index-1] = mat4(0.0);
    //    atomicAdd(StatusInt[0], -1);
    //    barrier();
    //}
}

void main(){
    if(particle_index < 4004002){
        AllocateParticles();
    }


}
