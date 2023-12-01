#version 460 core

layout(location=0) in int v_index; // vertex id
// out vec4 v_color; // color output

out GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
    vec4 vv_pos;
    vec4 vv_color;
    vec4 vvv_color;
}g_out;

// GeometryOutput{
//     // output vertex to demostrate all 3d-f, 2d-f, 3d-dvdt
//     vec4 v_pos;
//     vec4 v_color;
//     vec4 vv_pos;
//     vec4 vv_color;
//
// }g_out;

layout(std430, binding=0) buffer Particles{
    // particle inside domain with x, y, 0, voxel_id; ux, uy, 0, 0; vx, vy, 0, 0; aux, auy, avx, avy;
    // x        , y        , 0.0      , voxel_id ;
    // grad(n).x, grad(n).y, grad(v).x, grad(v).y;
    // lap(n)   , lap(v)   , n        , v        ;
    // dn/dt    , 0.0      , u.x      , u.y      ;
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

const float PI = 3.141592653589793;
const int n_voxel = 244824;
const float h = 0.05;
const float r = 0.005;
const int voxel_memory_length = 2912;
const int voxel_block_size = 960;
const float delta_t = 0.0000025;
const vec3 offset = vec3(-15.05, -0.05, -5.05);
const int VOXEL_GROUP_SIZE = 300000;
const float particle_volume = 8.538886859432597e-05;


uniform mat4 projection;
uniform mat4 view;

uniform int color_type;



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
    vec4 v_color;
    if(Particle[v_index][0].w != 0.0){
        // gl_Position = projection*view*vec4(Particle[v_index][0].xyz, 1.0); // set vertex position, w=1.0
        // int voxel_id = int(round(Particle[v_index][0].w));
        // vec3 voxel_center = vec3(float(Voxel[(voxel_id-1)*voxel_memory_length+1])*h, float(Voxel[(voxel_id-1)*voxel_memory_length+2])*h, float(Voxel[(voxel_id-1)*voxel_memory_length+3])*h);
        // float l = length(Particle[v_index][3].xyz);
        //v_color = vec4(abs(Particle[v_index][3].xyz), 1.0); // set output color by its acc
        //v_color = vec4(abs(sin(float(voxel_id/2))), abs(cos(float(voxel_id/3))), abs(sin(float(voxel_id/5))), 0.3);
        g_out.vv_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].w, Particle[v_index][0].z, 1.0);
        g_out.vv_color = vec4(Particle[v_index][3].y, -Particle[v_index][3].y, 0.0, 1.0);
        g_out.vvv_color = Particle[v_index][2].z==0.0 ? vec4(0.0): vec4(abs(sin(Particle[v_index][2].z)), abs(cos(Particle[v_index][2].z)), abs(sin(2*Particle[v_index][2].z)), 1.0);
        switch(color_type){
            case 0:  // n  0
                v_color = vec4(abs(sin(Particle[v_index][2].z)), abs(cos(Particle[v_index][2].z)), abs(sin(2*Particle[v_index][2].z)), 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].z, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].z, Particle[v_index][0].z, 1.0);
                break;
            case 1:  // dn/dt, red+green-  1
                v_color = vec4(Particle[v_index][3].x, -Particle[v_index][3].x, 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].z, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].z, Particle[v_index][0].z, 1.0);
                break;
            case 2:  // grad n  2
                v_color = vec4(abs(Particle[v_index][1].x), abs(Particle[v_index][1].y), 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, length(Particle[v_index][1].xy), Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, length(Particle[v_index][1].xy), Particle[v_index][0].z, 1.0);
                break;
            case 3:  // lap n  3
                v_color = vec4(Particle[v_index][2].x, -Particle[v_index][2].x, 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].x, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].x, Particle[v_index][0].z, 1.0);
                break;
            case 4:  // v  4
                v_color = vec4(abs(sin(Particle[v_index][2].w)), abs(cos(Particle[v_index][2].w)), abs(sin(2*Particle[v_index][2].w)), 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].w, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].w, Particle[v_index][0].z, 1.0);
                break;
            case 5:  // dv/dt  5
                v_color = vec4(Particle[v_index][3].y, -Particle[v_index][3].y, 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].w, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].w, Particle[v_index][0].z, 1.0);
                break;
            case 6:  // grad v  6
                v_color = vec4(abs(Particle[v_index][1].z), abs(Particle[v_index][1].w), 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, length(Particle[v_index][1].zw), Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, length(Particle[v_index][1].zw), Particle[v_index][0].z, 1.0);
                break;
            case 7:  // lap v 7
                v_color = vec4(Particle[v_index][2].y, -Particle[v_index][2].y, 0.0, 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, Particle[v_index][2].y, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, Particle[v_index][2].y, Particle[v_index][0].z, 1.0);
                break;
            case 8:  // u 8
                v_color = vec4(abs(sin(1000*length(Particle[v_index][3].zw))), abs(cos(1000*length(Particle[v_index][3].zw))), abs(sin(1000*2*length(Particle[v_index][3].zw))), 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, length(Particle[v_index][3].zw), Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, length(Particle[v_index][3].zw), Particle[v_index][0].z, 1.0);
                break;
            case 9:  // div(u) 9
                v_color = vec4(abs(sin(1000*ParticleSubData[v_index][0].x)), abs(cos(1000*ParticleSubData[v_index][0].x)), abs(sin(1000*2*ParticleSubData[v_index][0].x)), 1.0);
                // gl_Position = projection*view*vec4(Particle[v_index][0].x, ParticleSubData[v_index][0].x, Particle[v_index][0].z, 1.0);
                g_out.v_color = v_color;
                g_out.v_pos = vec4(Particle[v_index][0].x, ParticleSubData[v_index][0].x, Particle[v_index][0].z, 1.0);
                break;
                // x        , 0.0      , y        , voxel_id ;
                // grad(n).x, grad(n).y, grad(v).x, grad(v).y;
                // lap(n)   , lap(v)   , n        , v        ;
                // dn/dt    , dv/dt    , u.x      , u.y      ;

                // div(u), 0.0, 0.0, 0.0     ;
                // 0.0   , 0.0, 0.0, 0.0     ;
                // 0.0   , 0.0, 0.0, 0.0     ;
                // 0.0   , 0.0, 0.0, group_id;
        }
    }
    else{
        // gl_Position = vec4(0.0);
        v_color = vec4(0.0);
        g_out.v_color = v_color;
        g_out.v_pos = vec4(0.0);
        g_out.vv_pos = vec4(0);
        g_out.vv_color = v_color;
    }

}