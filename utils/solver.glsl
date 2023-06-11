#version 460 compatibility


layout(std430, binding=0) buffer Particals{
    mat4x4 Partical[];
};

layout(std430, binding=1) buffer BoundaryParticals{
    mat4x4 BoundaryPartical[];
};

layout(std430, binding=2) buffer Voxels{
    mat4x4 VoxelData[];
    // 1 voxel with 10 mat4x4
};

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint gid = gl_GlobalInvocationID.x;
int index = int(gid);

// function definitions

float rij;
float h;
vec3 xij;

const float PI = 3.141592653589793;


float h2 = h * h;

// coefficients
struct Coefficient{
    float Poly6_2d;
    float Poly6_3d;
    float Spiky_2d;
    float Spiky_3d;
    float Viscosity_2d;
    float Viscosity_3d;
};

Coefficient coeff = Coefficient(
    4 / (PI * pow(h, 8)),
    315 / (64 * PI * pow(h, 9)),
    10 / (PI * pow(h, 5)),
    15 / (PI * pow(h, 6)),
    40 / (PI * h2),
    15 / (2 * PI * pow(h, 3))
);


// poly6
float poly6_2d(float rij, float h){
    return max(0.0, coeff.Poly6_2d * pow((h2 - rij * rij),3));
}
float poly6_3d(float rij, float h){
    return max(0.0, coeff.Poly6_3d * pow((h2 - rij * rij),3));
}
vec2 grad_poly6_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = - 6 * coeff.Poly6_2d * pow((h2 - rij * rij),2);
    return vec2(w_prime * x, w_prime * y);
}
vec3 grad_poly6_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = - 6 * coeff.Poly6_3d * pow((h2 - rij * rij),2);
    return vec3(w_prime * x, w_prime * y, w_prime * z);
}
float lap_poly6_2d(float rij, float h){
    if (rij > h){return 0;}
    return - 12 * coeff.Poly6_2d * (h2 - rij * rij) * (h2 - 3 * rij * rij);
}
float lap_poly6_3d(float rij, float h){
    if (rij > h){return 0;}
    return - 6 * coeff.Poly6_3d * (h2 - rij * rij) * (3 * h2 - 7 * rij * rij);
}

// spiky
float spiky_2d(float rij, float h){
    return max(0.0, coeff.Spiky_2d * pow((h - rij),3));
}
float spiky_3d(float rij, float h){
    return max(0.0, coeff.Spiky_3d * pow((h - rij),3));
}
vec2 grad_spiky_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = - 3 * coeff.Spiky_2d * pow((h - rij),2);
    return vec2(w_prime * x / rij, w_prime * y / rij);
}
vec3 grad_spiky_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = - 3 * coeff.Spiky_3d * pow((h - rij),2);
    return vec3(w_prime * x / rij, w_prime * y / rij, w_prime * z / rij);
}
float lap_spiky_2d(float rij, float h){
    if (rij > h){return 0;}
    return coeff.Spiky_2d * (- 3 * h2 / rij + 12 * h - 9 * rij);
}
float lap_spiky_3d(float rij, float h){
    if (rij > h){return 0;}
    return coeff.Spiky_3d * (- 6 * h2 / rij + 18 * h - 12 * rij);
}

// viscosity
float viscosity_2d(float rij, float h){
    return max(0.0, coeff.Viscosity_2d * (- rij * rij * rij / (2 * h2) + rij * rij / h2 + h / (2 * rij) -1));
}
float viscosity_3d(float rij, float h){
    return max(0.0, coeff.Viscosity_3d * (- rij * rij * rij / (9 * h2) + rij * rij / (4 * h2) + log(rij / h) / 6 - 5 / 36));
}
vec2 grad_viscosity_2d(float x, float y, float rij, float h){
    if (rij > h){return vec2(0.0, 0.0);}
    float w_prime = coeff.Viscosity_2d * (- rij * rij / (3 * h2 * h) + rij / (2 * h2) - 1/ (6 * rij));
    return vec2(w_prime * x / rij, w_prime * y / rij);
}
vec3 grad_viscosity_3d(float x, float y, float z, float rij, float h){
    if (rij > h){return vec3(0.0, 0.0, 0.0);}
    float w_prime = coeff.Viscosity_3d * (- 3 * rij * rij / (h2 * h) + 2 * rij / h2 - h / (2 * rij * rij));
    return vec3(w_prime * x / rij, w_prime * y / rij, w_prime * z / rij);
}
float lap_viscosity_2d(float rij, float h){
    if (rij > h){return 0;}
    return 6 * coeff.Viscosity_2d / (h * h2) * (h - rij);
}
float lap_viscosity_3d(float rij, float h){
    if (rij > h){return 0;}
    return coeff.Viscosity_3d / (h * h2) * (h - rij);
}


/*
struct Particle{

    vec3 pos;
    vec3 vel;
    vec3 acc;
    float mass;
    float density;
    float pressure;

};
*/

void main() {
    float x = xij[0];
    float y = xij[1];
    float z = xij[2];
}
