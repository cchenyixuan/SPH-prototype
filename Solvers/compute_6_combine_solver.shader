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
const float MAX_PARTICLE_MASS = 6.545e-04;
const float ORIGINAL_PARTICLE_MASS = 6.545e-05;


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

void CombineToOne(){
    // mass_center = p = sum(m_i*p_i)/sum(m_i)
    vec3 mass_center = vec3(0.0);
    float sum_mass = 0.0;
    vec3 momentum = vec3(0.0);
    int current_particle_id;
    int combined_particle_id = 0;
    int combined_particle_voxel_slot_id;
    for(int i=0; i<voxel_block_size; ++i){
        current_particle_id = get_voxel_data(voxel_index, 32+i%voxel_block_size);
        // skip boundary particle
        if(current_particle_id<0){continue;}
        // empty slot encountered, all particles have been ckecked.
        if(current_particle_id==0){break;}
        mass_center += Particle[current_particle_id-1][1].w * Particle[current_particle_id-1][0].xyz;
        sum_mass += Particle[current_particle_id-1][1].w;
        momentum += Particle[current_particle_id-1][1].w * Particle[current_particle_id-1][1].xyz;
        // preserve combined_particle_id
        if(combined_particle_id==0){
            combined_particle_id = current_particle_id;  // start from 1
            combined_particle_voxel_slot_id = 32+i%voxel_block_size;
        }
        // erase this particle from system
            // set particle information to 0
        Particle[current_particle_id-1] = mat4(0.0);
        ParticleSubData[current_particle_id-1] = mat4(0.0);
            // total particle -1
        atomicAdd(StatusInt[0], -1);
        barrier();
        set_voxel_data(voxel_index, 32+i%voxel_block_size, 0);
    }
    mass_center /= sum_mass;
    vec3 velocity = momentum / sum_mass;
    // add new combined particle to current voxel
    Particle[combined_particle_id-1] = mat4(0.0);
    Particle[combined_particle_id-1][0] = vec4(mass_center.xyz, voxel_index_float);
    Particle[combined_particle_id-1][1] = vec4(velocity.xyz, sum_mass);
    ParticleSubData[combined_particle_id-1] = mat4(0.0);
    ParticleSubData[combined_particle_id-1][3].w = 3.0;

    set_voxel_data(voxel_index, combined_particle_voxel_slot_id, combined_particle_id);
    // total particle +1
    atomicAdd(StatusInt[0], 1);
    barrier();
}

void CombineNearBy(float threshold){
    /*
    Combine particles inside a voxel, merge 2 particles with distance less than threshold.
    One particle will be merged maximum one time in each loop.
    Merged particle will not have mass greater than 10*original-mass.
    */
    uint particle_1_id=0;
    uint particle_2_id=0;
    // check all particles inside voxel
    for(int i=0; i<voxel_block_size; ++i){
        // boundary or empty slot, contiune
        if(get_voxel_data(voxel_index, 32+i%voxel_block_size) <= 0){continue;}
        // actural particle encountered
        else{
            particle_1_id = get_voxel_data(voxel_index, 32+i%voxel_block_size);  // start from 1
            for(int j=i+1; j<voxel_block_size; ++j){
                // empty slot, contiune
                if(get_voxel_data(voxel_index, 32+j%voxel_block_size) <= 0){continue;}
                else{
                    particle_2_id = get_voxel_data(voxel_index, 32+j%voxel_block_size);  // start from 1
                    // particle distance < threshold, combine to mass center
                    if(Particle[particle_1_id-1][1].w+Particle[particle_2_id-1][1].w < MAX_PARTICLE_MASS && length(Particle[particle_1_id-1][0].xyz-Particle[particle_2_id-1][0].xyz)<threshold){
                        // new particle position = (m1*p1+m2*p2)/(m1+m2)
                        Particle[particle_1_id-1][0].xyz = (Particle[particle_1_id-1][0].xyz*Particle[particle_1_id-1][1].w + Particle[particle_2_id-1][0].xyz*Particle[particle_2_id-1][1].w)/(Particle[particle_1_id-1][1].w+Particle[particle_2_id-1][1].w);
                        // new particle velocity = (m1*v1+m2*v2)/(m1+m2)
                        Particle[particle_1_id-1][1].xyz = (Particle[particle_1_id-1][1].xyz*Particle[particle_1_id-1][1].w + Particle[particle_2_id-1][1].xyz*Particle[particle_2_id-1][1].w)/(Particle[particle_1_id-1][1].w+Particle[particle_2_id-1][1].w);
                        // new particle mass = m1+m2
                        Particle[particle_1_id-1][1].w = Particle[particle_1_id-1][1].w+Particle[particle_2_id-1][1].w;
                        // new particle phase = 3
                        ParticleSubData[particle_1_id-1][3].w = 3.0;
                        // clear particle 2 in Voxel and Particle
                        Particle[particle_2_id-1] = mat4(0.0);
                        ParticleSubData[particle_2_id-1] = mat4(0.0);
                        set_voxel_data(voxel_index, 32+j%voxel_block_size, 0);
                        VoxelParticleNumber[voxel_index-1] -= 1;
                        // set Total Particle number -= 1
                        atomicAdd(StatusInt[0], -1);
                        barrier();
                        break;
                    }
                }
            }
        }
    }
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
}

void SplitParticle(int insert_pointer){
    /*
    Split big particle with mass M into 2 parts:
    Small particle P1 with mass m; (m is original-mass)
    Large particle P2 with mass M-m.
    Large particle has index same as original particle,
    Small particle has new index and will be added to current voxel. (This split process requires Original particle not at boundary of current voxel)
    */
    // check every particle inside the voxel
    int particle_id;
    vec3 voxel_center = offset + vec3(float(get_voxel_data(voxel_index, 1))*h, float(get_voxel_data(voxel_index, 2))*h, float(get_voxel_data(voxel_index, 3))*h);
    for(int i=0; i<voxel_block_size; ++i){
        particle_id = get_voxel_data(voxel_index, 32+i%voxel_block_size);
        // skip boundary particle
        if(particle_id<0){continue;}
        // empty slot encountered, all particles have been ckecked.
        if(particle_id==0){break;}
        // if this particle can be splited (big mass and not nearby boundary)
        if(Particle[particle_id-1][1].w>ORIGINAL_PARTICLE_MASS && length(Particle[particle_id-1][0].xyz-voxel_center)<0.4*h){
            // Smaller One
            int new_particle_index = atomicAdd(StatusInt[1], 1);  // start from 0
            barrier();
            atomicAdd(StatusInt[0], 1);  // total + 1
            barrier();
            Particle[new_particle_index] = mat4(0.0);  // can be skipped
            ParticleSubData[new_particle_index] = mat4(0.0);  // can be skipped
            // position = Original-position - velo-direction*(M-m)/(M)*2r
            Particle[new_particle_index][0].xyz = Particle[particle_id-1][0].xyz + normalize(Particle[particle_id-1][1].xyz)*((Particle[particle_id-1][1].w-ORIGINAL_PARTICLE_MASS)/Particle[particle_id-1][1].w)*r*0.001;
            // voxel id is same
            Particle[new_particle_index][0].w = Particle[particle_id-1][0].w;
            // mass is m
            Particle[new_particle_index][1].w = ORIGINAL_PARTICLE_MASS;
            // velo is same
            Particle[new_particle_index][1].xyz = Particle[particle_id-1][1].xyz;
            // phase is same
            ParticleSubData[new_particle_index][3].w = 4.0;
            // handel voxel info
            set_voxel_data(voxel_index, insert_pointer, new_particle_index+1);
            VoxelParticleNumber[voxel_index-1] += 1;
            insert_pointer += 1;

            // Bigger One
            // position = Original-position + velo-direction*(m)/(M)*2r
            Particle[particle_id-1][0].xyz -= normalize(Particle[particle_id-1][1].xyz)*(ORIGINAL_PARTICLE_MASS/Particle[particle_id-1][1].w)*r*0.001;
            // mass -= ORIGINAL_PARTICLE_MASS
            Particle[particle_id-1][1].w -= ORIGINAL_PARTICLE_MASS;

        }
    }
}

void SelfAdaptiveParticle(){
    // get max velocity inside the voxel
    float max_velocity_magnitude = 0.0;
    // float max_type=0.0;
    bool empty_voxel=true;
    int current_particle_id;
    int insert_ptr;
    for(int i=0; i<voxel_block_size; ++i){
        current_particle_id = get_voxel_data(voxel_index, 32+i%voxel_block_size);
        // skip boundary particle
        if(current_particle_id<0){continue;}
        // empty slot encountered, all particles have been ckecked.
        if(current_particle_id==0){insert_ptr=32+i%voxel_block_size; break;}
        empty_voxel = false;
        max_velocity_magnitude = max(max_velocity_magnitude, length(Particle[current_particle_id-1][1].xyz));
        // max_type = max(max_type, ParticleSubData[current_particle_id-1][3].w);
    }
    // combine if max_velocity_magnitude < threshold
    if(max_velocity_magnitude<0.1 && !empty_voxel){
        CombineNearBy(0.9*r);
    }
    // split if max_velocity_magnitude > threshold
    else if(max_velocity_magnitude>0.2 && !empty_voxel){
        SplitParticle(insert_ptr);
    }
}

void main() {
    SelfAdaptiveParticle();
}
