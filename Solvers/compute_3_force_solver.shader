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
int particle_index = int(gid)+1;
float particle_index_float = float(particle_index);

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
const float Coeff_Poly6_2d = 1.0;  // 4 / (PI * pow(h, 8));
const float Coeff_Poly6_3d = 1.0;  // 315 / (64 * PI * pow(h, 9));
const float Coeff_Spiky_2d = 1.0;  // 10 / (PI * pow(h, 5));
const float Coeff_Spiky_3d = 1.0;  // 15 / (PI * pow(h, 6));
const float Coeff_Viscosity_2d = 1.0; // 40 / (PI * h2);
const float Coeff_Viscosity_3d = 1.0; // 15 / (2 * PI * pow(h, 3));


float h2 = h * h;

// coefficients
// struct Coefficient{
//     float Poly6_2d;
//     float Poly6_3d;
//     float Spiky_2d;
//     float Spiky_3d;
//     float Viscosity_2d;
//     float Viscosity_3d;
// };
//
// Coefficient coeff = Coefficient(
//     4 / (PI * pow(h, 8)),
//     315 / (64 * PI * pow(h, 9)),
//     10 / (PI * pow(h, 5)),
//     15 / (PI * pow(h, 6)),
//     40 / (PI * h2),
//     15 / (2 * PI * pow(h, 3))
// );


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
// Wendland C4
float wendland_3d(float rij, float h){
    float q = rij/h;
    if (q > 1){return 0;}
    return Coeff_Wendland_3d * pow(1-q, 6)*(35/3*q*q+6*q+1);
}
vec3 grad_wendland_3d(float x, float y, float z, float rij, float h){
    float q = rij/h;
    if (q > 1){return vec3(0.0);}
    float w_prime = Coeff_Wendland_3d / h * (-56/3) * q * (1+5*q) * pow(1-q, 5);
    return w_prime * vec3(x / rij, y / rij, z / rij);
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

vec3 GetExternalForce(vec3 pos){
    // vec3 gravity = vec3(0.0, -9.81, 0.0);
    vec3 force = vec3(0.0, 0.0, 0.0);
    if(pos.x<0.2){
        vec3 center = vec3(0.27, 0.34, pos.z);

        vec3 po = center-pos;
        vec3 right = vec3(0.0, 0.0, 1.0);
        force = normalize(cross(po, right))*2.0;
    }
    else{
        force = vec3(0.0, 0.0, 0.0);
    }



    return force;
}



void ComputeParticleForce(){
    // position of current particle focused
    vec3 particle_pos = Particle[particle_index-1][0].xyz;
    // voxel_id of current particle
    int voxel_id = int(round(Particle[particle_index-1][0].w));  // starts from 1
    // empty f_pressure, f_viscosity, f_external
    vec3 a_pressure = vec3(0.0, 0.0, 0.0);
    vec3 a_viscosity = vec3(0.0, 0.0, 0.0);
    vec3 a_external = vec3(0.0, -9.81, 0.0);  // gravity
    vec3 f_cohesion = vec3(0.0, 0.0, 0.0);  // surface tension of domain particles
    vec3 f_adhesion = vec3(0.0, 0.0, 0.0);  // surface tension of boundary particles
    // vec3 f_transfer = vec3(0.0, 0.0, 0.0);  // Vorticity transfer force
    // vec3 t_transfer = vec3(0.0, 0.0, 0.0);  // Vorticity transfer torque
    vec3 kernel_tmp = vec3(0.0);
    // find neighbourhood vertices, i.e., P_j
    // search in same voxel
    // calculate vertices inside
    for (int j=0; j<voxel_block_size; ++j){
        // vertex index
        int index_j = get_voxel_data(voxel_id, 32+j);  // starts from 1
        if (index_j==0){ break; }// empty slot
        if (particle_index==index_j){ continue; }
        // P_j is a domain particle
        if (index_j>0){
            // vector xij
            vec3 xij = particle_pos - Particle[index_j-1][0].xyz;
            // distance rij
            float rij = length(xij);
            // distance less than h
            if (rij<h){
                kernel_tmp = grad_wendland_3d(xij.x, xij.y, xij.z, rij, h);
                // add f_pressure and f_viscosity
                // a_press -= grad_spiky_3d(xij, rij, H)*(MASS_j*(P_j_pressure/P_j_rho**2 + P_i_pressure/P_i_rho**2))
                a_pressure -= kernel_tmp * (Particle[index_j-1][1].w*(Particle[index_j-1][3].w/pow(Particle[index_j-1][2].w, 2) + Particle[particle_index-1][3].w/pow(Particle[particle_index-1][2].w, 2)));
                // f_visco  += VISC * (         P_j_mass        /         P_j_rho         ) * (        P_j_velocity       -            P_i_velocity          ) * lap_viscosity_3d(rij, h)
                // f_viscosity += VISC * (Particle[index_j-1][1].w / Particle[index_j-1][2].w) * (Particle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz) * lap_viscosity_3d(rij, h);
                //f_viscosity += Particle[particle_index-1][1].w*VISC*(Particle[index_j-1][1].w/Particle[index_j-1][2].z)*(Particle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz)*(2*length(grad_spiky_3d(xij.x, xij.y, xij.z, rij, h))/rij);
                // a_visco  += VISC * 2*(dimension+2) * MASS_j/P_j_rho * (P_i_v - P_j_v)*(P_i_x-P_j_x)/(rij*rij + 0.01*H**2) * grad_spiky_3d(xij, rij, H)
                a_viscosity += viscosity* 10 * (Particle[index_j-1][1].w/Particle[index_j-1][2].w) * dot(Particle[particle_index-1][1].xyz-Particle[index_j-1][1].xyz, Particle[particle_index-1][0].xyz-Particle[index_j-1][0].xyz)/(rij*rij+0.01*h2) * kernel_tmp;
                // f_cohesion -= COHESION * MASS_j*(P_i_x-P_j_x)*poly6_3d(rij, h);
                f_cohesion -= cohesion * Particle[index_j-1][1].w*xij*poly6_3d(rij, h);

                // f_transfer += MASS_i*VISC_TRANSFER * 1/rho_i * (P_i_v_angluar-P_j_v_angluar)xgrad_spiky_3d(xij.x, xij.y, xij.z, rij, h);  // VISC_TRANSFER refers to mu/rho
                // f_transfer += Particle[particle_index-1][1].w*VISC_TRANSFER/Particle[particle_index-1][2].w * cross((Particle[particle_index-1][2].xyz-Particle[index_j-1][2].xyz), grad_spiky_3d(xij.x, xij.y, xij.z, rij, h));
                // t_transfer += MASS_i*VISC_TRANSFER * 1/rho_i * (P_i_v-P_j_v)xgrad_spiky_3d(xij.x, xij.y, xij.z, rij, h);  // this term needs to -2*P_i_v_angluar afrterwards
                // t_transfer += Particle[particle_index-1][1].w*VISC_TRANSFER/Particle[particle_index-1][2].w * cross((Particle[particle_index-1][1].xyz-Particle[index_j-1][1].xyz), grad_spiky_3d(xij.x, xij.y, xij.z, rij, h));

            }
        }
        // P_j is a boundary particle
        else if (index_j<0){
            // reverse index_j
            index_j = abs(index_j);
            // vector xij
            vec3 xij = particle_pos - BoundaryParticle[index_j-1][0].xyz;
            // distance rij
            float rij = length(xij);
            // distance less than h
            if (rij<h){
                kernel_tmp = grad_wendland_3d(xij.x, xij.y, xij.z, rij, h);
                // add f_pressure and f_viscosity
                // a_press -= grad_spiky_3d(xij, rij, H)*(MASS_j*(P_j_pressure/P_j_rho**2 + P_i_pressure/P_i_rho**2))
                a_pressure -= kernel_tmp * (BoundaryParticle[index_j-1][1].w*(BoundaryParticle[index_j-1][3].w/pow(BoundaryParticle[index_j-1][2].w, 2) + Particle[particle_index-1][3].w/pow(Particle[particle_index-1][2].w, 2)));
                // f_visco  += VISC * (             P_j_mass            /             P_j_rho             ) * (            P_j_velocity           -            P_i_velocity          ) * lap_viscosity_3d(rij, h)
                // f_viscosity += VISC * (BoundaryParticle[index_j-1][1].w / BoundaryParticle[index_j-1][2].w) * (BoundaryParticle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz) * lap_viscosity_3d(rij, h);
                //f_viscosity += Particle[particle_index-1][1].w*VISC*(BoundaryParticle[index_j-1][1].w/BoundaryParticle[index_j-1][2].z)*(BoundaryParticle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz)*(2*length(grad_spiky_3d(xij.x, xij.y, xij.z, rij, h))/rij);
                // a_visco  += VISC * 2*(dimension+2) * MASS_j/P_j_rho * (P_i_v - P_j_v)*(P_i_x-P_j_x)/(rij*rij + 0.01*H**2) * grad_spiky_3d(xij, rij, H)
                a_viscosity += viscosity* 10 * (BoundaryParticle[index_j-1][1].w/BoundaryParticle[index_j-1][2].w) * dot(Particle[particle_index-1][1].xyz-BoundaryParticle[index_j-1][1].xyz, Particle[particle_index-1][0].xyz-BoundaryParticle[index_j-1][0].xyz)/(rij*rij+0.01*h2) * kernel_tmp;
                // f_adhesion -= ADHESION * MASS_j*(P_i_x-P_j_x)*poly6_3d(rij, h);
                f_adhesion -= adhesion * BoundaryParticle[index_j-1][1].w*xij*poly6_3d(rij, h);
            }
        }

    }

    // search in neighbourhood voxels
    for(int i=4; i<30; ++i){
        // its neighbourhood voxel
        int neighborhood_id = get_voxel_data(voxel_id, i);
        // valid neighborhood
        if(neighborhood_id!=0){
            // calculate vertices inside
            for (int j=0; j<voxel_block_size; ++j){
                // vertex index
                int index_j = get_voxel_data(neighborhood_id, 32+j);  // starts from 1
                if (index_j==0){ break; }// empty slot
                if (particle_index==index_j){ continue; }
                // P_j is a domain particle
                if (index_j>0){
                    // vector xij
                    vec3 xij = particle_pos - Particle[index_j-1][0].xyz;
                    // distance rij
                    float rij = length(xij);
                    // distance less than h
                    if (rij<h){
                        kernel_tmp = grad_wendland_3d(xij.x, xij.y, xij.z, rij, h);
                        // add f_pressure and f_viscosity
                        // a_press -= grad_spiky_3d(xij, rij, H)*(MASS_j*(P_j_pressure/P_j_rho**2 + P_i_pressure/P_i_rho**2))
                        a_pressure -= kernel_tmp * (Particle[index_j-1][1].w*(Particle[index_j-1][3].w/pow(Particle[index_j-1][2].w, 2) + Particle[particle_index-1][3].w/pow(Particle[particle_index-1][2].w, 2)));
                        // f_visco  += VISC * (         P_j_mass        /         P_j_rho         ) * (        P_j_velocity       -            P_i_velocity          ) * lap_viscosity_3d(rij, h)
                        // f_viscosity += VISC * (Particle[index_j-1][1].w / Particle[index_j-1][2].w) * (Particle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz) * lap_viscosity_3d(rij, h);
                        //f_viscosity += Particle[particle_index-1][1].w*VISC*(Particle[index_j-1][1].w/Particle[index_j-1][2].z)*(Particle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz)*(2*length(grad_spiky_3d(xij.x, xij.y, xij.z, rij, h))/rij);
                        // a_visco  += VISC * 2*(dimension+2) * MASS_j/P_j_rho * (P_i_v - P_j_v)*(P_i_x-P_j_x)/(rij*rij + 0.01*H**2) * grad_spiky_3d(xij, rij, H)
                        a_viscosity += viscosity* 10 * (Particle[index_j-1][1].w/Particle[index_j-1][2].w) * dot(Particle[particle_index-1][1].xyz-Particle[index_j-1][1].xyz, Particle[particle_index-1][0].xyz-Particle[index_j-1][0].xyz)/(rij*rij+0.01*h2) * kernel_tmp;
                        // f_cohesion -= COHESION * MASS_j*(P_i_x-P_j_x)*poly6_3d(rij, h);
                        f_cohesion -= cohesion * Particle[index_j-1][1].w*xij*poly6_3d(rij, h);

                        // f_transfer += MASS_i*VISC_TRANSFER * 1/rho_i * (P_i_v_angluar-P_j_v_angluar)xgrad_spiky_3d(xij.x, xij.y, xij.z, rij, h);  // VISC_TRANSFER refers to mu/rho
                        // f_transfer += Particle[particle_index-1][1].w*VISC_TRANSFER/Particle[particle_index-1][2].w * cross((Particle[particle_index-1][2].xyz-Particle[index_j-1][2].xyz), grad_spiky_3d(xij.x, xij.y, xij.z, rij, h));
                        // t_transfer += MASS_i*VISC_TRANSFER * 1/rho_i * (P_i_v-P_j_v)xgrad_spiky_3d(xij.x, xij.y, xij.z, rij, h);  // this term needs to -2*P_i_v_angluar afrterwards
                        // t_transfer += Particle[particle_index-1][1].w*VISC_TRANSFER/Particle[particle_index-1][2].w * cross((Particle[particle_index-1][1].xyz-Particle[index_j-1][1].xyz), grad_spiky_3d(xij.x, xij.y, xij.z, rij, h));

                    }
                }
                // P_j is a boundary particle
                else if (index_j<0){
                    // reverse index_j
                    index_j = abs(index_j);
                    // vector xij
                    vec3 xij = particle_pos - BoundaryParticle[index_j-1][0].xyz;
                    // distance rij
                    float rij = length(xij);
                    // distance less than h
                    if (rij<h){
                        kernel_tmp = grad_wendland_3d(xij.x, xij.y, xij.z, rij, h);
                        // add f_pressure and f_viscosity
                        // a_press -= grad_spiky_3d(xij, rij, H)*(MASS_j*(P_j_pressure/P_j_rho**2 + P_i_pressure/P_i_rho**2))
                        a_pressure -= kernel_tmp * (BoundaryParticle[index_j-1][1].w*(BoundaryParticle[index_j-1][3].w/pow(BoundaryParticle[index_j-1][2].w, 2) + Particle[particle_index-1][3].w/pow(Particle[particle_index-1][2].w, 2)));
                        // f_visco  += VISC * (             P_j_mass            /             P_j_rho             ) * (            P_j_velocity           -            P_i_velocity          ) * lap_viscosity_3d(rij, h)
                        // f_viscosity += VISC * (BoundaryParticle[index_j-1][1].w / BoundaryParticle[index_j-1][2].w) * (BoundaryParticle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz) * lap_viscosity_3d(rij, h);
                        //f_viscosity += Particle[particle_index-1][1].w*VISC*(BoundaryParticle[index_j-1][1].w/BoundaryParticle[index_j-1][2].z)*(BoundaryParticle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz)*(2*length(grad_spiky_3d(xij.x, xij.y, xij.z, rij, h))/rij);
                        // a_visco  += VISC * 2*(dimension+2) * MASS_j/P_j_rho * (P_i_v - P_j_v)*(P_i_x-P_j_x)/(rij*rij + 0.01*H**2) * grad_spiky_3d(xij, rij, H)
                        a_viscosity += viscosity* 10 * (BoundaryParticle[index_j-1][1].w/BoundaryParticle[index_j-1][2].w) * dot(Particle[particle_index-1][1].xyz-BoundaryParticle[index_j-1][1].xyz, Particle[particle_index-1][0].xyz-BoundaryParticle[index_j-1][0].xyz)/(rij*rij+0.01*h2) * kernel_tmp;
                        // f_adhesion -= ADHESION * MASS_j*(P_i_x-P_j_x)*poly6_3d(rij, h);
                        f_adhesion -= adhesion * BoundaryParticle[index_j-1][1].w*xij*poly6_3d(rij, h);

                        // f_visco = m_i*mu*lap(v_i)
                        //lap_v_i  = sum{m_i*mu * m_j/rho_j * (vi-vj) * 2||(lap_W(ij))||/||rij||}
                        // f_viscosity += m_i*VISC*(m_j/rho_j)*(vj-vi)*(2*lap_viscosity_3d(rij, h)/rij);
                        // f_viscosity += Particle[particle_index-1][1].w*VISC*(Particle[index_j-1][1].w/Particle[index_j-1][2].z)*(Particle[index_j-1][1].xyz - Particle[particle_index-1][1].xyz)*(2*lap_viscosity_3d(rij, h)/rij);
                    }
                }

            }

        }
    }

    // compute force
    //            P_i_acceleration           = (f_pressure + f_viscosity + f_external)/mass
    // t_transfer -= Particle[particle_index-1][1].w*VISC_TRANSFER*2*Particle[particle_index-1][2].xyz;
    // ParticleSubData[particle_index-1][0].xyz = t_transfer;
    Particle[particle_index-1][3].xyz = a_pressure + a_viscosity + a_external + (f_cohesion + f_adhesion)/Particle[particle_index-1][1].w;
}

void main() {
    if(Particle[particle_index-1][0].w != 0){
        ComputeParticleForce();
    }

}
