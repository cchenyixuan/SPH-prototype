#version 460 core

layout (points) in;
layout (line_strip, max_vertices = 2) out;
in GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
}g_in[];
out vec4 v_color;

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


uniform mat4x4 projection;
uniform mat4x4 view;


void CreateVector(){
    // 2 vertices
    vec4 p1 = g_in[0].v_pos;
    vec4 p2 = vec4(g_in[0].v_pos.xyz+g_in[0].v_color.xyz, 1.0);
    // vertex color
    v_color = g_in[0].v_color;
    // 2 vertices emittion to generate a vector line
    gl_Position = projection*view*p1;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    // end of line-strip
    EndPrimitive();
}

void main() {
    CreateVector();

}
