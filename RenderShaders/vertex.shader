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


const float PI = 3.141592653589793;
const int n_boundary_particle = 10383;
const int n_voxel = 48235;
const float h = 0.01;
const float r = 0.0025;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float rest_dense = 1000.0;
const float eos_constant = 32142.0;
const float delta_t = 5e-05;
const float viscosity = 0.0001;
const float cohesion = 0.0001;
const float adhesion = 0.0001;
const vec3 offset = vec3(-0.434871, -0.690556, -0.245941);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 6.545e-08;

uniform mat4 projection;
uniform mat4 view;

uniform int color_type;

int v_index = gl_InstanceID;

vec3 get_color_gradient(float ratio, float range){
    // ratio in range 0.9-1.1
    /*
        0.9 --> purple(0.5, 0.0, 1.0)
        0.93 --> blue(0.0, 0.0, 1.0)
        0.96 --> cyan(0.0, 1.0, 1.0)
        1.0 --> green(0.0, 1.0, 0.0)
        1.03 --> yellow(1.0, 1.0, 0.0)
        1.06 --> orange(1.0, 0.5, 0.0)
        1.1 --> red(1.0, 0.0, 0.0)
    */
    float red = 0.0;
    if(ratio<1.0){red=min(max(-15*(ratio-(1.0-range*2/3)), 0.0), 1.0);}
    else if(ratio>=1.0){red=min(max(3/range*(ratio-1.0), 0.0), 1.0);}
    float green = 0.0;
    if(ratio<1.0){green=min(max(30*(ratio-(1.0-range*2/3)), 0.0), 1.0);}
    else if(ratio>=1.0){green=min(max(-1.5/range*(ratio-(1.0+range)), 0.0), 1.0);}
    float blue = min(max(-3/range*(ratio-1.0), 0.0), 1.0);
    return vec3(red, green, blue);
}

void main() {
    if(Particle[v_index][0].w > 0.0){
        gl_Position = projection*view*vec4(Particle[v_index][0].xyz + sphere_pos*r, 1.0); // set vertex position, w=1.0
        gl_PointSize = 4.0;
        // int voxel_id = int(round(Particle[v_index][0].w));
        // vec3 voxel_center = vec3(float(Voxel[(voxel_id-1)*voxel_memory_length+1])*h, float(Voxel[(voxel_id-1)*voxel_memory_length+2])*h, float(Voxel[(voxel_id-1)*voxel_memory_length+3])*h);
        // float l = length(Particle[v_index][3].xyz);
        //v_color = vec4(abs(Particle[v_index][3].xyz), 1.0); // set output color by its acc
        //v_color = vec4(abs(sin(float(voxel_id/2))), abs(cos(float(voxel_id/3))), abs(sin(float(voxel_id/5))), 0.3);
        switch(color_type){
            case 0:  // velocity
                v_color = vec4(abs(normalize(Particle[v_index][1].xyz)), 1.0);
                // v_color = vec4(1.7671e-09*ParticleSubData[v_index][2].x, 1.7671e-09*ParticleSubData[v_index][2].x, 1.7671e-09*ParticleSubData[v_index][2].x, 1.0);
                break;
            case 1:  // acc
                v_color = vec4(abs(normalize(Particle[v_index][3].xyz)), 1.0);
                break;
            case 2:  // pressure(density)
                v_color = vec4(get_color_gradient(abs(Particle[v_index][2].w)/rest_dense, 0.01).xyz, 1.0);
                break;
            case 3:  // N phase
                if(ParticleSubData[v_index][3].w==1.0){v_color = vec4(0.0, 1.0, 0.0, 1.0);}
                else if(ParticleSubData[v_index][3].w==2.0){v_color = vec4(1.0, 1.0, 0.0, 1.0);}
                else if(ParticleSubData[v_index][3].w==3.0){v_color = vec4(0.0, 1.0, 1.0, 1.0);}
                else if(ParticleSubData[v_index][3].w==4.0){v_color = vec4(0.0, 0.0, 1.0, 1.0);}
                else{v_color = vec4(1.0, 0.0, 0.0, 1.0);}
                break;
            case 4:  // kernel value
                v_color = vec4(get_color_gradient(ParticleSubData[v_index][2].x, 0.1).xyz, 1.0);
                break;
            case 5:  // d_rho/dt
                v_color = vec4(get_color_gradient(abs(ParticleSubData[v_index][3].x)/float(StatusInt[2]), 0.1).xyz, 1.0);
                break;
            case 6:  // rho
                v_color = vec4(get_color_gradient(Particle[v_index][2].w/1000.0, 0.1).xyz, 1.0);
                break;
            case 7:  // curl
                v_color = vec4(abs(ParticleSubData[v_index][1].xyz)/ParticleSubData[v_index][3].z, 1.0);
                break;
            case 8:  // temperature
                v_color = vec4(get_color_gradient(abs(Particle[v_index][2].z)/30.0, 0.1).xyz, 0.7);
                break;
            case 9:  // DT
                v_color = vec4(Particle[v_index][2].y, -Particle[v_index][2].y, 0.0, 0.7);
                break;
        }
    }
    else{
        // this particle is not exist, pass
        ;
    }

}