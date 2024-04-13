#version 460 compatibility

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
layout(std430, binding=6) coherent buffer GlobalStatus{
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

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

uint x_length = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
uint y_length = gl_NumWorkGroups.y * gl_WorkGroupSize.y;
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*x_length + gl_GlobalInvocationID.z*x_length*y_length;
int particle_index = int(gid)+1;
float particle_index_float = float(particle_index);

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

// sample code--------------
// const int Inlet1ParticleNumber = StatusInt[3];
// const int Inlet2ParticleNumber = StatusInt[4];
// const int Inlet3ParticleNumber = StatusInt[5];
//
// int Inlet1In = StatusInt[9];
// int Inlet2In = StatusInt[10];
// int Inlet3In = StatusInt[11];
//
// int Inlet1Pointer = StatusInt[6];
// int Inlet2Pointer = StatusInt[7];
// int Inlet3Pointer = StatusInt[8];
// -------------------------
float h2 = h * h;

int get_voxel_data(int voxel_id, int pointer){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    int ans;
    switch(voxel_buffer_index){
        case 0:
            ans = Voxel0[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 1:
            ans = Voxel1[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 2:
            ans = Voxel2[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 3:
            ans = Voxel3[voxel_local_index*voxel_memory_length+pointer];
            break;
        case 4:
            ans = Voxel4[voxel_local_index*voxel_memory_length+pointer];
            break;
        //case 5:
        //    ans = Voxel5[voxel_local_index*voxel_memory_length+pointer];
        //    break;
    }
    return ans;
}

void set_voxel_data(int voxel_id, int pointer, int value){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    value: value to set
    return: value before changed
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    switch(voxel_buffer_index){
        case 0:
            Voxel0[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 1:
            Voxel1[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 2:
            Voxel2[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 3:
            Voxel3[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        case 4:
            Voxel4[voxel_local_index*voxel_memory_length+pointer] = value;
            break;
        //case 5:
        //    Voxel5[voxel_local_index*voxel_memory_length+pointer] = value;
        //    break;
    }
}

int set_voxel_data_atomic(int voxel_id, int pointer, int value){
    /*
    voxel_id: starts from 1
    pointer: from 0 to voxel_memory_length
    value: value to set
    return: value before changed
    */
    int voxel_buffer_index = (voxel_id-1) / VOXEL_GROUP_SIZE;
    int voxel_local_index = (voxel_id-1) - voxel_buffer_index*VOXEL_GROUP_SIZE;
    int ans;
    switch(voxel_buffer_index){
        case 0:
            ans = atomicAdd(Voxel0[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 1:
            ans = atomicAdd(Voxel1[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 2:
            ans = atomicAdd(Voxel2[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 3:
            ans = atomicAdd(Voxel3[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        case 4:
            ans = atomicAdd(Voxel4[voxel_local_index*voxel_memory_length+pointer], value);
            break;
        //case 5:
        //    ans = atomicAdd(Voxel5[voxel_local_index*voxel_memory_length+pointer], value);
        //    break;
    }
    return ans;
}

void arrange_voxel_particle_out(int flag, int voxel_id){
    int destination_voxel_pointer;
    switch (flag){
        // left -100
        case -100:
            destination_voxel_pointer = 4;
            break;
        // right 100
        case 100:
            destination_voxel_pointer = 5;
            break;
        // down -10
        case -10:
            destination_voxel_pointer = 6;
            break;
        // up 10
        case 10:
            destination_voxel_pointer = 7;
            break;
        // back -1
        case -1:
            destination_voxel_pointer = 8;
            break;
        // front 1
        case 1:
            destination_voxel_pointer = 9;
            break;
        // left_down -110
        case -110:
            destination_voxel_pointer = 10;
            break;
        // left_up -90
        case -90:
            destination_voxel_pointer = 11;
            break;
        // right_down 90
        case 90:
            destination_voxel_pointer = 12;
            break;
        // right_up 110
        case 110:
            destination_voxel_pointer = 13;
            break;
        // left_back -101
        case -101:
            destination_voxel_pointer = 14;
            break;
        // left_front -99
        case -99:
            destination_voxel_pointer = 15;
            break;
        // right_back 99
        case 99:
            destination_voxel_pointer = 16;
            break;
        // right_front 101
        case 101:
            destination_voxel_pointer = 17;
            break;
        // down_back -11
        case -11:
            destination_voxel_pointer = 18;
            break;
        // down_front -9
        case -9:
            destination_voxel_pointer = 19;
            break;
        // up_back 9
        case 9:
            destination_voxel_pointer = 20;
            break;
        // up_front 11
        case 11:
            destination_voxel_pointer = 21;
            break;
        // left_down_back -111
        case -111:
            destination_voxel_pointer = 22;
            break;
        // left_down_front -109
        case -109:
            destination_voxel_pointer = 23;
            break;
        // left_up_back -91
        case -91:
            destination_voxel_pointer = 24;
            break;
        // left_up_front -89
        case -89:
            destination_voxel_pointer = 25;
            break;
        // right_down_back 89
        case 89:
            destination_voxel_pointer = 26;
            break;
        // right_down_front 91
        case 91:
            destination_voxel_pointer = 27;
            break;
        // right_up_back 109
        case 109:
            destination_voxel_pointer = 28;
            break;
        // right_up_front 111
        case 111:
            destination_voxel_pointer = 29;
            break;

    }
    int destination_voxel_id = get_voxel_data(voxel_id, destination_voxel_pointer);  // starts from 1, if is 0: this particle should vanish
    //
    if(destination_voxel_id>0){
        int i = atomicAdd(VoxelParticleInNumber[destination_voxel_id-1], 1);
        barrier();
        set_voxel_data_atomic(destination_voxel_id, 32+voxel_block_size+voxel_block_size+i%voxel_block_size, particle_index);  // starts from 1 (domain particle)
        barrier();
        // set particle vertex_id to the new one
        Particle[particle_index-1][0].w = float(destination_voxel_id);
    }
    else{
        // set particle information to 0
        Particle[particle_index-1] = mat4(0.0);
        ParticleSubData[particle_index-1] = mat4(0.0);
        // total particle -1
        atomicAdd(StatusInt[0], -1);
        barrier();
    }
}

void arrange_inlet_particle(){
    // Inlet1In+Inlet2In+Inlet3In>0
    bool IsOccupied = false;
    int ParticleShouldGenerate;
    ParticleShouldGenerate = atomicAdd(StatusInt[9], -1);
    barrier();
    int ParticleEjectIndex;
    if(IsOccupied==false){
        if (ParticleShouldGenerate>0){
            IsOccupied = true;
            // new particle is a copy of Inlet1[Inlet1Pointer]

            ParticleEjectIndex = atomicAdd(StatusInt[6], 1);
            barrier();
            // avoid eject 2 particle at same location
            ParticleEjectIndex %= StatusInt[3];

            Particle[particle_index-1]= Inlet1[ParticleEjectIndex];
            ParticleSubData[particle_index-1] = mat4(0.0);
            ParticleSubData[particle_index-1][3].w = 1.0;
            // voxel update
            int destination_voxel_id = int(round(Particle[particle_index-1][0].w));// starts from 1
            int i = atomicAdd(VoxelParticleInNumber[destination_voxel_id-1], 1);
            barrier();
            set_voxel_data_atomic(destination_voxel_id, 32+voxel_block_size+voxel_block_size+i%voxel_block_size, particle_index);// starts from 1 (domain particle)
            barrier();

        }
        else {
            StatusInt[9] = 0;
        }
    }
    if(IsOccupied==false){
        ParticleShouldGenerate = atomicAdd(StatusInt[10], -1);
        barrier();
        if(ParticleShouldGenerate>0){
            IsOccupied = true;
            // new particle is a copy of Inlet2[Inlet2Pointer]
            ParticleEjectIndex = atomicAdd(StatusInt[7], 1);
            barrier();
            // avoid eject 2 particle at same location
            ParticleEjectIndex %= StatusInt[4];

            Particle[particle_index-1]= Inlet2[ParticleEjectIndex];
            ParticleSubData[particle_index-1] = mat4(0.0);
            ParticleSubData[particle_index-1][3].w = 2.0;
            // voxel update
            int destination_voxel_id = int(round(Particle[particle_index-1][0].w));  // starts from 1
            int i = atomicAdd(VoxelParticleInNumber[destination_voxel_id-1], 1);
            barrier();
            set_voxel_data_atomic(destination_voxel_id, 32+voxel_block_size+voxel_block_size+i%voxel_block_size, particle_index);  // starts from 1 (domain particle)
            barrier();

        }
        else{
            StatusInt[10] = 0;
        }
    }
    if(IsOccupied==false){
        ParticleShouldGenerate = atomicAdd(StatusInt[11], -1);
        barrier();
        if(ParticleShouldGenerate>0){
            IsOccupied = true;
            // new particle is a copy of Inlet2[Inlet2Pointer]
            ParticleEjectIndex = atomicAdd(StatusInt[8], 1);
            barrier();
            // avoid eject 2 particle at same location
            ParticleEjectIndex %= StatusInt[5];

            Particle[particle_index-1]= Inlet3[ParticleEjectIndex];
            ParticleSubData[particle_index-1] = mat4(0.0);
            ParticleSubData[particle_index-1][3].w = 3.0;
            // voxel update
            int destination_voxel_id = int(round(Particle[particle_index-1][0].w));  // starts from 1
            int i = atomicAdd(VoxelParticleInNumber[destination_voxel_id-1], 1);
            barrier();
            set_voxel_data_atomic(destination_voxel_id, 32+voxel_block_size+voxel_block_size+i%voxel_block_size, particle_index);  // starts from 1 (domain particle)
            barrier();

        }
        else{
            StatusInt[11] = 0;
        }
    }
    if(IsOccupied==true){
        // StatusInt[0] += 1;
        atomicAdd(StatusInt[0], 1);
        barrier();
        // maximum particle index should be upgrade
        atomicMax(StatusInt[1], particle_index);
        barrier();
    }

}

void EulerMethod(){
    // current voxel id
    int voxel_id = int(round(Particle[particle_index-1][0].w));  // starts from 1
    if(voxel_id==0){
        // inlet handled here
        arrange_inlet_particle();
    }
    else{
        // rho += d_rho/dt * dt
        Particle[particle_index-1][2].w += delta_t*ParticleSubData[particle_index-1][3].x;
        Particle[particle_index-1][2].w = max(0.0, Particle[particle_index-1][2].w);
        // Particle[particle_index-1][2].w = ParticleSubData[particle_index-1][3].y;

        // p = EOS(rho)
        // Particle[particle_index-1][3].w = max(eos_constant * (pow(Particle[particle_index-1][2].w/rest_dense, 7) -1), 0.0);
        // calculate future position
        //   move =             P_velocity           *   dt   +      P_acceleration  *     dt/2
        vec3 move = Particle[particle_index-1][1].xyz*delta_t + Particle[particle_index-1][3].xyz*delta_t*delta_t/2;
        // estimate future position
        //   future_pos =             P_position            + move
        vec3 future_pos = Particle[particle_index-1][0].xyz + move;
        // current voxel center
        vec3 voxel_center = offset + vec3(float(get_voxel_data(voxel_id, 1))*h, float(get_voxel_data(voxel_id, 2))*h, float(get_voxel_data(voxel_id, 3))*h);
        // x axis flag (-1: left, 0: current, 1: right)
        int x_axis_flag;
        if     (voxel_center.x+h/2<=future_pos.x){x_axis_flag=1;}
        else if(voxel_center.x-h/2<=future_pos.x && future_pos.x<voxel_center.x+h/2){x_axis_flag=0;}
        else if(future_pos.x<voxel_center.x-h/2){x_axis_flag=-1;}
        // y axis flag (-1: below, 0: current, 1: above)
        int y_axis_flag;
        if     (voxel_center.y+h/2<=future_pos.y){y_axis_flag=1;}
        else if(voxel_center.y-h/2<=future_pos.y && future_pos.y<voxel_center.y+h/2){y_axis_flag=0;}
        else if(future_pos.y<voxel_center.y-h/2){y_axis_flag=-1;}
        // z axis flag (-1: back, 0: current, 1: front)
        int z_axis_flag;
        if     (voxel_center.z+h/2<=future_pos.z){z_axis_flag=1;}
        else if(voxel_center.z-h/2<=future_pos.z && future_pos.z<voxel_center.z+h/2){z_axis_flag=0;}
        else if(future_pos.z<voxel_center.z-h/2){z_axis_flag=-1;}
        // identify which voxel the particle will go
        int flag = x_axis_flag*100 + y_axis_flag*10 + z_axis_flag;
        // remain inside 0
        if (flag==0){
            // do nothing
        }
        // particle goes to other voxel
        else {
            // set voxel out buffer
            // o starts from 0 and max is voxel_block_size-1, this will add 1 to o and return o before addition happends
            int o = atomicAdd(VoxelParticleOutNumber[voxel_id-1], 1);
            barrier();
            set_voxel_data_atomic(voxel_id, 32+voxel_block_size+o%voxel_block_size, particle_index);  // starts from 1 (domain particle)
            barrier();
            arrange_voxel_particle_out(flag, voxel_id);
        }

        // particle position and velocity will be set and particle acceleration will be erased
        if(Particle[particle_index-1][0].w>0){
            Particle[particle_index-1][0].xyz = future_pos;
            Particle[particle_index-1][1].xyz += Particle[particle_index-1][3].xyz*delta_t;
            // Particle[particle_index-1][2].xyz += ParticleSubData[particle_index-1][0].xyz*DELTA_T/(2*Particle[particle_index-1][1].w);
        }
    }

}

void main() {
    EulerMethod();
}
