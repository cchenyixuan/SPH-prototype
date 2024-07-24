#version 460 core

layout(location=0) in int v_index; // vertex id
out GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
}g_out;


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
    if(Particle[v_index][0].w >= 0.0){
        g_out.v_pos = vec4(Particle[v_index][0].xyz, 1.0); // set vertex position, w=1.0
        switch(vector_type){
            case 0:  // velocity
                g_out.v_color = vec4(Particle[v_index][1].xyz/length(Particle[v_index][1].xyz)*r, 1.0); // set vertex color use velo, w=1.0
                break;
            case 1:  // acceleration
                g_out.v_color = vec4(Particle[v_index][3].xyz/length(Particle[v_index][3].xyz)*r, 1.0); // set vertex color use acc, w=1.0
                break;
            // case2 clear all domain particle visibility
            case 2:
                g_out.v_color = vec4(0.0);
                break;
            case 9:  // normal
                g_out.v_color = vec4(ParticleSubData[v_index][1].xyz/length(ParticleSubData[v_index][1].xyz)*2*r, length(ParticleSubData[v_index][1].xyz));
                break;
            // new shader same as above, named boundary vector vertex shader, change particle to boundaryparticle
            // rewrite g_out.v_color with pressure and cirection at BP[id][3].xyzw
        }
    }
    else{
        g_out.v_pos = vec4(0.0);
        g_out.v_color = vec4(0.0);
    }

}
