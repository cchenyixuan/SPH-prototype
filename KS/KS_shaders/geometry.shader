#version 460 core

layout (points) in;
layout (points, max_vertices = 5) out;
in GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
    vec4 vv_pos;
    vec4 vv_color;
    vec4 vvv_color;
}g_in[];
out vec4 v_color;

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

void CreateCross(){

}

void main() {
    // origin
    v_color = g_in[0].v_color;
    vec4 v_pos1 = g_in[0].v_pos;
    gl_Position = projection*view*v_pos1;
    EmitVertex();
    EndPrimitive();
    // 2d-version
    v_color = g_in[0].v_color;
    vec4 v_pos2 = vec4(g_in[0].v_pos.x+35, g_in[0].v_pos.z, 0.0, 1.0);
    gl_Position = projection*view*v_pos2;
    EmitVertex();
    EndPrimitive();
    // dv/dt
    v_color = g_in[0].vv_color;
    vec4 v_pos3 = vec4(g_in[0].vv_pos.x-35, g_in[0].vv_pos.yzw);
    gl_Position = projection*view*v_pos3;
    EmitVertex();
    EndPrimitive();
    // dv/dt 2d version
    v_color = g_in[0].vv_color;
    vec4 v_pos4 = vec4(g_in[0].vv_pos.x-70, g_in[0].vv_pos.z, 0.0, 1.0);
    gl_Position = projection*view*v_pos4;
    EmitVertex();
    EndPrimitive();
    // u 2d version colored by |u|
    v_color = g_in[0].vvv_color;
    vec4 v_pos5 = vec4(g_in[0].v_pos.x+70, g_in[0].v_pos.z, 0.0, 1.0);
    gl_Position = projection*view*v_pos5;
    EmitVertex();
    EndPrimitive();

}
