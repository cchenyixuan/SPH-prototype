#version 460 core

layout (points) in;
layout (line_strip, max_vertices = 2) out;
in GeometryOutput{
    vec4 v_pos;
    vec4 v_color;
}g_in[];
out vec4 v_color;

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
