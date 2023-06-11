#version 460 compatibility

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

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;
int voxel_index = int(gid)+1;
float voxel_index_float = float(voxel_index);

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

void UpgradeVoxel(){
    // create a counter to check all out buffer has been considered
    int out_counter = 0;
    for (int i=0; i<voxel_block_size; ++i){
        // read out buffer, out buffer is continuous
        int out_particle_id = get_voxel_data(voxel_index, 32+voxel_block_size+i%voxel_block_size);  // starts from 1
        // if zero found, iteration should stop
        if (out_particle_id==0){ break; }
        for (int j=0; j<voxel_block_size; ++j){
            // read inside buffer and check if particle id match
            int inside_particle_id = get_voxel_data(voxel_index, 32+j%voxel_block_size);
            if (inside_particle_id==out_particle_id){
                // erase voxel particle slot and add counter
                set_voxel_data(voxel_index, 32+j%voxel_block_size, 0);
                out_counter += 1;
                // break
                break;
            }
        }
    }
    // check if out_counter == VoxelParticleOutNumber[voxel_index-1]
    if (out_counter==VoxelParticleOutNumber[voxel_index-1]){
        // verified
    }
    else {
        // particle number not match! debug here
    }
    // create a counter to check all in buffer has been considered
    int in_counter = 0;
    for (int i=0; i<voxel_block_size; ++i){
        // read in buffer, in buffer is continuous
        int in_particle_id = get_voxel_data(voxel_index, 32+voxel_block_size+voxel_block_size+i%voxel_block_size);  // starts from 1
        // if zero found, iteration should stop
        if (in_particle_id==0){ break; }
        for (int j=0; j<voxel_block_size; ++j){
            // read inside buffer and check if a slot is found
            if (get_voxel_data(voxel_index, 32+j%voxel_block_size)==0){
                // insert voxel particle slot and add counter
                set_voxel_data(voxel_index, 32+j%voxel_block_size, in_particle_id);
                in_counter += 1;
                // break
                break;
            }
        }
    }
    // check if in_counter == VoxelParticleInNumber[voxel_index-1]
    if (in_counter==VoxelParticleInNumber[voxel_index-1]){
        // verified
    }
    else {
        // particle number not match! debug here
    }
    // code above could be modified to have O(n) time complexity
    // re-arrange inside buffer using 2 pointers
    int ptr1 = 0;// slow
    int ptr2 = 0;// fast
    while (ptr2<voxel_block_size){
        if (get_voxel_data(voxel_index, 32+ptr2%voxel_block_size)==0){
            // empty slot found
            ptr2 += 1;
        }
        else if (get_voxel_data(voxel_index, 32+ptr2%voxel_block_size)!=0 && ptr1!=ptr2){
            // filled slot found
            set_voxel_data(voxel_index, 32+ptr1%voxel_block_size, get_voxel_data(voxel_index, 32+ptr2%voxel_block_size));
            set_voxel_data(voxel_index, 32+ptr2%voxel_block_size, 0);
            ptr1 += 1;
        }
        else {
            ptr1 += 1;
            ptr2 += 1;
        }
    }
    // above code persists points order
    // following code breaks points order but could have better performance
    /*
int ptr1 = 0;
int ptr2 = 95;
while(ptr1<ptr2){
    if(Voxel[(voxel_index-1)*voxel_memory_length+32+ptr1%voxel_block_size]==0){
        if(Voxel[(voxel_index-1)*voxel_memory_length+32+ptr2%voxel_block_size]!=0){
            Voxel[(voxel_index-1)*voxel_memory_length+32+ptr1%voxel_block_size] = Voxel[(voxel_index-1)*voxel_memory_length+32+ptr2%voxel_block_size];
            Voxel[(voxel_index-1)*voxel_memory_length+32+ptr2%voxel_block_size] = 0;
        }
        else{
            ptr2 -= 1;
        }
    }
    else{
        ptr1 += 1;
    }
}
*/

    // all out particles have been checked, clear out buffer
    for (int i=0; i<voxel_block_size; ++i){
        set_voxel_data(voxel_index, 32+voxel_block_size+i%voxel_block_size, 0);
    }
    // all in particles have been checked, clear in buffer
    for (int i=0; i<voxel_block_size; ++i){
        set_voxel_data(voxel_index, 32+voxel_block_size+voxel_block_size+i%voxel_block_size, 0);
    }
    // re-calculate VoxelParticleNumber and clear VoxelParticleInNumber and VoxelParticleOutNumber
    VoxelParticleNumber[voxel_index-1] += -VoxelParticleOutNumber[voxel_index-1]+VoxelParticleInNumber[voxel_index-1];
    VoxelParticleOutNumber[voxel_index-1] = 0;
    VoxelParticleInNumber[voxel_index-1] = 0;
}

void main() {
    UpgradeVoxel();
}
