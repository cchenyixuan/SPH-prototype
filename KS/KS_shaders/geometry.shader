#version 460 core

layout (points) in;
layout (points, max_vertices = 2) out;
in GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
    vec4 vv_pos;
    vec4 vv_color;
}g_in[];
out vec4 v_color;

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


uniform mat4 projection;
uniform mat4 view;

void CreateCube(){
    // center of voxel
    vec4 center = g_in[0].v_pos;
    // 8 vertices
    vec4 p4 = vec4(center.x-h/2, center.y-h/2, center.z-h/2, 1.0);
    vec4 p3 = vec4(center.x+h/2, center.y-h/2, center.z-h/2, 1.0);
    vec4 p8 = vec4(center.x+h/2, center.y-h/2, center.z+h/2, 1.0);
    vec4 p7 = vec4(center.x-h/2, center.y-h/2, center.z+h/2, 1.0);
    vec4 p6 = vec4(center.x-h/2, center.y+h/2, center.z+h/2, 1.0);
    vec4 p2 = vec4(center.x-h/2, center.y+h/2, center.z-h/2, 1.0);
    vec4 p1 = vec4(center.x+h/2, center.y+h/2, center.z-h/2, 1.0);
    vec4 p5 = vec4(center.x+h/2, center.y+h/2, center.z+h/2, 1.0);

    // vertex color
    v_color = g_in[0].v_color;
    // 12 vertices emittion to generate a full cube in order 4-3-7-8-5-3-1-4-2-7-6-5-2-1
    gl_Position = projection*view*p4;
    EmitVertex();
    gl_Position = projection*view*p3;
    EmitVertex();
    gl_Position = projection*view*p7;
    EmitVertex();
    gl_Position = projection*view*p8;
    EmitVertex();
    gl_Position = projection*view*p5;
    EmitVertex();
    gl_Position = projection*view*p3;
    EmitVertex();
    gl_Position = projection*view*p1;
    EmitVertex();
    gl_Position = projection*view*p4;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    gl_Position = projection*view*p7;
    EmitVertex();
    gl_Position = projection*view*p6;
    EmitVertex();
    gl_Position = projection*view*p5;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    gl_Position = projection*view*p1;
    EmitVertex();

    // end of triangle-strip
    EndPrimitive();


}

void main() {
    v_color = g_in[0].v_color;
    vec4 v_pos1 = g_in[0].v_pos;
    gl_Position = projection*view*v_pos1;
    EmitVertex();
    v_color = g_in[0].vv_color;
    vec4 v_pos2 = g_in[0].vv_pos+vec4(20, 0.0, 0.0, 0.0);
    gl_Position = projection*view*v_pos2;
    EmitVertex();
    EndPrimitive();

}