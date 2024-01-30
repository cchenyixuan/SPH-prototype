#version 460 compatibility

layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, 0, voxel_id; ux, uy, 0, 0; vx, vy, 0, 0; aux, auy, avx, avy;
    // x    , y        , 0.0, voxel_id
    // u    , lap(u)   , F1 , 0.0
    // u+F1 , lap(u+F1), F2 , 0.0
    // du   , 0.0      , 0.0, 0.0
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
const int n_voxel = 15376;
const float h = 0.1;
const float r = 0.005;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float delta_t = 0.000005;
const vec3 offset = vec3(-3.100777,  -0.108929,  -3.0821009);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 0.00010305493648345673;


float h2 = h*h;
float Coeff_Wendland_2d = 9 / (PI * pow(h, 2));


// Wendland C4
float wendland_2d(float rij, float h){
    float q = rij/h;
    if (q > 1){return 0.0;}
    return Coeff_Wendland_2d * pow(1-q, 6)*(35/3*q*q+6*q+1);
}
vec2 grad_wendland_2d(float x, float y, float rij, float h){
    float q = rij/h;
    if (q > 1){return vec2(0.0);}
    float w_prime = Coeff_Wendland_2d / h * (-56/3) * q * (1+5*q) * pow(1-q, 5);
    return w_prime * vec2(x / rij, y / rij);
}

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

void ComputeParticleProperties(){
    // DEBUG FOR KERNEL VALUE
    float kernel_value = 0.0;
    vec2 kernel_tmp;
    // position of current particle focused
    vec2 particle_pos = Particle[particle_index-1][0].xz;
    // delete its grad and laplacian last time, optional
    // Particle[particle_index-1][1] = vec4(0.0);
    Particle[particle_index-1][2].yzw = vec3(0.0);
    Particle[particle_index-1][3] = vec4(0.0);
    // voxel_id of current particle
    int voxel_id = int(round(Particle[particle_index-1][0].w));  // starts from 1
    // find neighbourhood vertices, i.e., P_j
    // search in same voxel
    // calculate vertices inside
    for (int j=0; j<voxel_block_size; ++j){
        // vertex index
        int index_j = get_voxel_data(voxel_id, 32+j);  // starts from 1 or -1
        if (index_j==0){break;}// empty slot
        else if (index_j>=358802){break;}// empty slot
        // P_j is a domain particle
        else if (index_j>0){

            // distance rij
            float rij = distance(particle_pos, Particle[index_j-1][0].xz);
            vec2 xij = particle_pos - Particle[index_j-1][0].xz;

            // distance less than h
            if (particle_index == index_j){continue;}
            else if (rij<0.1*r){continue;}
            else if (rij<h){
                // kernel
                kernel_tmp = grad_wendland_2d(xij.x, xij.y, rij, h);
                // lap u
                Particle[particle_index-1][2].y -= particle_volume * (Particle[particle_index-1][2].x-Particle[index_j-1][2].x) * 2 * length(kernel_tmp)/(rij);
                // Particle[particle_index-1][2].y += 8*particle_volume * (Particle[particle_index-1][2].x-Particle[index_j-1][2].x) * dot(xij, kernel_tmp)/(rij*rij + 0.01*h2);
            }
        }

    }

    // search in neighbourhood voxels
    for(int i=4; i<30; ++i){
        // its neighbourhood voxel
        int neighborhood_id = get_voxel_data(voxel_id, i);  // starts from 1
        // valid neighborhood
        if(neighborhood_id!=0){
            // calculate vertices inside
            for (int j=0; j<voxel_block_size; ++j){
                // vertex index
                int index_j = get_voxel_data(neighborhood_id, 32+j);  // starts from 1 or -1
                if (index_j==0){ break; }// empty slot
                else if (index_j>=358802){break;}// empty slot
                // P_j is a domain particle
                else if (index_j>0){
                    // distance rij
                    float rij = distance(particle_pos, Particle[index_j-1][0].xz);
                    vec2 xij = particle_pos - Particle[index_j-1][0].xz;
                    // distance less than h
                    if (particle_index == index_j){continue;}
                    else if (rij<0.1*r){continue;}
                    else if (rij<h){
                        // kernel
                        kernel_tmp = grad_wendland_2d(xij.x, xij.y, rij, h);
                        // lap u
                        Particle[particle_index-1][2].y -= particle_volume * (Particle[particle_index-1][2].x-Particle[index_j-1][2].x) * 2 * length(kernel_tmp)/(rij);
                        // Particle[particle_index-1][2].y += 8*particle_volume * (Particle[particle_index-1][2].x-Particle[index_j-1][2].x) * dot(xij, kernel_tmp)/(rij*rij + 0.01*h2);
                    }
                }

            }

        }
    }
    // L^-1
    Particle[particle_index-1][2].y *= 1.0;
    // F2
    Particle[particle_index-1][2].z = delta_t*Particle[particle_index-1][2].y;
    //du = (F1+F2)/2
    Particle[particle_index-1][3].x = 0.5*(Particle[particle_index-1][1].z+Particle[particle_index-1][2].z);
    // u += (F1+F2)/2
    Particle[particle_index-1][1].x += Particle[particle_index-1][3].x;
}

void main() {
    if(Particle[particle_index-1][0].w > 0.5){
        ComputeParticleProperties();
    }

}
