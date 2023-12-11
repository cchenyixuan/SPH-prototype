#version 460 compatibility

layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, 0, voxel_id; ux, uy, 0, 0; vx, vy, 0, 0; aux, auy, avx, avy;
    // x , y , 0 , voxel_id
    // grad(u).x, grad(u).y, grad(v).x, grad(v).y
    // lap(u), lap(v), u, v
    // du/dt.x, du/dt.y, 0.0, 0.0
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
const int n_voxel = 646416;
const float h = 0.05;
const float r = 0.005;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float delta_t = 0.0000025;
const vec3 offset = vec3(-10.05, -0.05, -10.05);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 8.538886859432597e-05;


float h2 = h*h;
float Coeff_Poly6_2d = 4 / (PI * pow(h, 8));
float Coeff_Poly6_3d = 315 / (64 * PI * pow(h, 9));
float Coeff_Spiky_2d = 10 / (PI * pow(h, 5));
float Coeff_Spiky_3d = 15 / (PI * pow(h, 6));
float Coeff_Viscosity_2d = 40 / (PI * pow(h, 2));
float Coeff_Viscosity_3d = 15 / (2 * PI * pow(h, 3));


// poly6
float poly6_2d(float rij, float h){
    return max(0.0, Coeff_Poly6_2d * pow((h2 - rij * rij),3));
}
float poly6_3d(float rij, float h){
    return max(0.0, Coeff_Poly6_3d * pow((h2 - rij * rij),3));
}
vec2 grad_poly6_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = - 6 * Coeff_Poly6_2d * pow((h2 - rij * rij),2);
    return vec2(w_prime * x, w_prime * y);
}
vec3 grad_poly6_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = - 6 * Coeff_Poly6_3d * pow((h2 - rij * rij),2);
    return vec3(w_prime * x, w_prime * y, w_prime * z);
}
float lap_poly6_2d(float rij, float h){
    if (rij > h){return 0;}
    return - 12 * Coeff_Poly6_2d * (h2 - rij * rij) * (h2 - 3 * rij * rij);
}
float lap_poly6_3d(float rij, float h){
    if (rij > h){return 0;}
    return - 6 * Coeff_Poly6_3d * (h2 - rij * rij) * (3 * h2 - 7 * rij * rij);
}

// spiky
float spiky_2d(float rij, float h){
    return max(0.0, Coeff_Spiky_2d * pow((h - rij),3));
}
float spiky_3d(float rij, float h){
    return max(0.0, Coeff_Spiky_3d * pow((h - rij),3));
}
vec2 grad_spiky_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = - 3 * Coeff_Spiky_2d * pow((h - rij),2);
    if(rij == 0){return vec2(0.0, 0.0);}
    return vec2(w_prime * x / rij, w_prime * y / rij);
}
vec3 grad_spiky_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = - 3 * Coeff_Spiky_3d * pow((h - rij),2);
    return vec3(w_prime * x / rij, w_prime * y / rij, w_prime * z / rij);
}
float lap_spiky_2d(float rij, float h){
    if (rij > h){return 0;}
    return Coeff_Spiky_2d * (- 3 * h2 / rij + 12 * h - 9 * rij);
}
float lap_spiky_3d(float rij, float h){
    if (rij > h){return 0;}
    return Coeff_Spiky_3d * (- 6 * h2 / rij + 18 * h - 12 * rij);
}

// viscosity
float viscosity_2d(float rij, float h){
    return max(0.0, Coeff_Viscosity_2d * (- rij * rij * rij / (2 * h2) + rij * rij / h2 + h / (2 * rij) -1));
}
float viscosity_3d(float rij, float h){
    return max(0.0, Coeff_Viscosity_3d * (- rij * rij * rij / (9 * h2) + rij * rij / (4 * h2) + log(rij / h) / 6 - 5 / 36));
}
vec2 grad_viscosity_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = Coeff_Viscosity_2d * (- rij * rij / (3 * h2 * h) + rij / (2 * h2) - 1/ (6 * rij));
    return vec2(w_prime * x / rij, w_prime * y / rij);
}
vec3 grad_viscosity_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = Coeff_Viscosity_3d * (- 3 * rij * rij / (h2 * h) + 2 * rij / h2 - h / (2 * rij * rij));
    return vec3(w_prime * x / rij, w_prime * y / rij, w_prime * z / rij);
}
float lap_viscosity_2d(float rij, float h){
    if (rij > h){return 0;}
    return 6 * Coeff_Viscosity_2d / (h * h2) * (h - rij);
}
float lap_viscosity_3d(float rij, float h){
    if (rij > h){return 0;}
    return Coeff_Viscosity_3d / (h * h2) * (h - rij);
}
vec2 grad_log_2d(float x, float y, float rij, float h){
    // if (rij > h){return vec2(0.0, 0.0);}
    return vec2(x / (10000000.0*rij*rij), y / (10000000.0*rij*rij));
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
    Particle[particle_index-1][1] = vec4(0.0);
    Particle[particle_index-1][2].xy = vec2(0.0);
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
        else if (index_j>=4004002){break;}// empty slot
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
                kernel_tmp = grad_spiky_2d(xij.x, xij.y, rij, h);
                // kernel_value += particle_volume*lap_viscosity_2d(rij, h);
                // grad u
                Particle[particle_index-1][1].xy += particle_volume*(-Particle[particle_index-1][2].z+Particle[index_j-1][2].z) * kernel_tmp;
                // grad u^(q-1)
                // Particle[particle_index-1][1].xy += particle_volume*(-Particle[particle_index-1][2].z^(q-1)+Particle[index_j-1][2].z^(q-1)) * kernel_tmp;
                // grad v
                //Particle[particle_index-1][1].zw += particle_volume*(-Particle[particle_index-1][2].w+Particle[index_j-1][2].w) * kernel_tmp;
                Particle[particle_index-1][1].zw += particle_volume*(-Particle[particle_index-1][2].w+Particle[index_j-1][2].w) * kernel_tmp;

                // lap u
                Particle[particle_index-1][2].x += 8 * particle_volume * (Particle[particle_index-1][2].z-Particle[index_j-1][2].z) * dot(xij, kernel_tmp)/(rij*rij);
                // lap (u^m)
                // Particle[particle_index-1][2].x += 8 * particle_volume * (Particle[particle_index-1][2].z^m-Particle[index_j-1][2].z^m) * dot(xij, kernel_tmp)/(rij*rij);
                // lap v
                Particle[particle_index-1][2].y -= particle_volume * (Particle[particle_index-1][2].w-Particle[index_j-1][2].w) * 2 * length(kernel_tmp)/(rij);

                // LPSPH v
                // ParticleSubData[particle_index-1][0].y += (2*log(0.005213455767878209)-1)*0.005213455767878209*0.005213455767878209/2 * Particle[index_j-1][2].z;

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
                else if (index_j>=4004002){break;}// empty slot
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
                        kernel_tmp = grad_spiky_2d(xij.x, xij.y, rij, h);
                        //kernel_value += particle_volume*lap_viscosity_2d(rij, h);
                        // grad u
                        Particle[particle_index-1][1].xy += particle_volume*(-Particle[particle_index-1][2].z+Particle[index_j-1][2].z)* kernel_tmp;
                        // grad v
                        //Particle[particle_index-1][1].zw += particle_volume*(-Particle[particle_index-1][2].w+Particle[index_j-1][2].w)* kernel_tmp;
                        Particle[particle_index-1][1].zw += particle_volume*(-Particle[particle_index-1][2].w+Particle[index_j-1][2].w) * kernel_tmp;
                        // lap u
                        Particle[particle_index-1][2].x += 8 * particle_volume * (Particle[particle_index-1][2].z-Particle[index_j-1][2].z) * dot(xij, kernel_tmp)/(rij*rij);
                        // lap v
                        Particle[particle_index-1][2].y -= particle_volume * (Particle[particle_index-1][2].w-Particle[index_j-1][2].w) * 2 * length(kernel_tmp)/(rij);

                        // LPSPH v
                        // ParticleSubData[particle_index-1][0].y += (2*log(0.005213455767878209)-1)*0.005213455767878209*0.005213455767878209/2 * Particle[index_j-1][2].z;
                    }
                }

            }

        }
    }
    // Particle[particle_index-1][2].x /= kernel_value;
    //Particle[particle_index-1][2].y /= kernel_value;
    //grad v(x) = (N*u)(x) = -1/(2*PI) * integal((x-y)/(x-y)**2 * u(y)dy)
    // for (int j=0; j<5000000; ++j){
    //     if (Particle[j][0].w == 0.0){
    //         continue;
    //     }
    //     else if(particle_index-1 == j){
    //         continue;
    //     }
    //     else{
    //         float rij = distance(particle_pos, Particle[j][0].xz);
    //         vec2 xij = particle_pos - Particle[j][0].xz;
    //         Particle[particle_index-1][1].zw += Particle[j][2].z*particle_volume*grad_log_2d(xij.x, xij.y, rij, h);
    //     }
    // }
    // Particle[particle_index-1][1].zw *= -10000000.0/(2*PI);
    // Particle[particle_index-1][2].y = Particle[particle_index-1][2].w-Particle[particle_index-1][2].z;
    // debug lap(v)-v+u=0
    // Particle[particle_index-1][3].z = Particle[particle_index-1][2].y+Particle[particle_index-1][2].z;
    // Particle[particle_index-1][3].w = Particle[particle_index-1][2].w-Particle[particle_index-1][3].z;
    // Particle[particle_index-1][2].y = -Particle[particle_index-1][2].z;
}

void main() {
    if(Particle[particle_index-1][0].w > 0.5){
        ComputeParticleProperties();
    }

}
