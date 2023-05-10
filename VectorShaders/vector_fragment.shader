#version 460 core


in vec4 v_color; // color output
out vec4 o_color;

void main() {
    // set output color to v_color
    o_color = vec4(abs(v_color.xyz)/length(v_color.xyz), 1.0);
}