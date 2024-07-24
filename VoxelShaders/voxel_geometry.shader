#version 460 core

layout (points) in;
// layout (triangle_strip, max_vertices = 14) out;
layout (line_strip, max_vertices = 24) out;
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

uniform mat4 projection;
uniform mat4 view;

void CreateCube(){
    // center of voxel
    vec4 center = g_in[0].v_pos;
    // 8 vertices
    vec4 p4 = vec4(center.x-h/2, center.y-h/2, center.z-h/2, 1.0);  // 1
    vec4 p3 = vec4(center.x+h/2, center.y-h/2, center.z-h/2, 1.0);  // 2
    vec4 p8 = vec4(center.x+h/2, center.y-h/2, center.z+h/2, 1.0);  // 3
    vec4 p7 = vec4(center.x-h/2, center.y-h/2, center.z+h/2, 1.0);  // 4
    vec4 p6 = vec4(center.x-h/2, center.y+h/2, center.z+h/2, 1.0);  // 5
    vec4 p2 = vec4(center.x-h/2, center.y+h/2, center.z-h/2, 1.0);  // 6
    vec4 p1 = vec4(center.x+h/2, center.y+h/2, center.z-h/2, 1.0);  // 7
    vec4 p5 = vec4(center.x+h/2, center.y+h/2, center.z+h/2, 1.0);  // 8

    // vertex color
    v_color = g_in[0].v_color;
    /*
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
    */
    // 1-2-3-4 5-6-7-8 1-2-7-6 3-4-5-8 1-4-5-6 2-3-8-7
    gl_Position = projection*view*p4;
    EmitVertex();
    gl_Position = projection*view*p3;
    EmitVertex();
    gl_Position = projection*view*p8;
    EmitVertex();
    gl_Position = projection*view*p7;
    EmitVertex();
    EndPrimitive();

    gl_Position = projection*view*p6;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    gl_Position = projection*view*p1;
    EmitVertex();
    gl_Position = projection*view*p5;
    EmitVertex();
    EndPrimitive();

    gl_Position = projection*view*p4;
    EmitVertex();
    gl_Position = projection*view*p3;
    EmitVertex();
    gl_Position = projection*view*p1;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    EndPrimitive();

    gl_Position = projection*view*p8;
    EmitVertex();
    gl_Position = projection*view*p7;
    EmitVertex();
    gl_Position = projection*view*p6;
    EmitVertex();
    gl_Position = projection*view*p5;
    EmitVertex();
    EndPrimitive();

    gl_Position = projection*view*p4;
    EmitVertex();
    gl_Position = projection*view*p7;
    EmitVertex();
    gl_Position = projection*view*p6;
    EmitVertex();
    gl_Position = projection*view*p2;
    EmitVertex();
    EndPrimitive();

    gl_Position = projection*view*p8;
    EmitVertex();
    gl_Position = projection*view*p3;
    EmitVertex();
    gl_Position = projection*view*p1;
    EmitVertex();
    gl_Position = projection*view*p5;
    EmitVertex();
    EndPrimitive();



}

void main() {
    CreateCube();

}
