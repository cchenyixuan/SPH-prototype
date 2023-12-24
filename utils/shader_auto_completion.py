import re
import os


export_dir = "./runtime"
os.makedirs(export_dir, exist_ok=True)

shader_src_list = [
    "RenderShaders/boundary_vertex.shader",
    "RenderShaders/fragment.shader",
    "RenderShaders/vertex.shader",
    "Solvers/compute_0_init_boundary_particles.shader",
    "Solvers/compute_1_init_domain_particles.shader",
    "Solvers/compute_1_init_inlet_particles.shader",
    "Solvers/compute_2_boundary_density_pressure_solver.shader",
    "Solvers/compute_2_density_pressure_solver.shader",
    "Solvers/compute_2_density_derivative_solver.shader",
    "Solvers/compute_3_force_solver.shader",
    "Solvers/compute_4_integrate_solver.shader",
    "Solvers/compute_5_voxel_upgrade_solver.shader",
    "Solvers/compute_6_combine_solver.shader",
    "VectorShaders/vector_vertex.shader",
    "VectorShaders/vector_geometry.shader",
    "VectorShaders/vector_fragment.shader",
    "VectorShaders/vector_boundary_vertex.shader",
    "VoxelShaders/voxel_vertex.shader",
    "VoxelShaders/voxel_geometry.shader",
    "VoxelShaders/voxel_fragment.shader",
    "VoxelShaders/voxel_compute.shader",
    "Solvers_WSSD/wssd_compute_1_init_domain_particles.shader",
    "Solvers_WSSD/wssd_compute_2_density_pressure_solver.shader",
    "Solvers_WSSD/wssd_compute_3_force_solver.shader",
]
shader_export_list = [
    export_dir+"/"+shader.split("/")[-1] for shader in shader_src_list
]


find_const = re.compile(r"(const .*;\n)", re.S)
const_variables = {
    "PI": lambda value: f"const float PI = {float(value)};\n",
    "n_boundary_particle": lambda value: f"const int n_boundary_particle = {int(value)};\n" if abs(int(value)-value) < 1e-8 else print(f"n_boundary_particle is an integer, while provided with another type, {type(value)}!"),
    "n_voxel": lambda value: f"const int n_voxel = {int(value)};\n" if abs(int(value)-value) < 1e-8 else print(f"n_voxel is an integer, while provided with another type, {type(value)}!"),
    "h": lambda value: f"const float h = {float(value)};\n",
    "r": lambda value: f"const float r = {float(value)};\n",
    "voxel_memory_length": lambda value: f"const int voxel_memory_length = {int(value)};\n" if abs(int(value)-value) < 1e-8 else print(f"voxel_memory_length is an integer, while provided with another type, {type(value)}!"),
    "voxel_block_size": lambda value: f"const int voxel_block_size = {int(value)};\n" if abs(int(value)-value) < 1e-8 else print(f"voxel_block_size is an integer, while provided with another type, {type(value)}!"),
    "rest_dense": lambda value: f"const float rest_dense = {float(value)};\n",
    "eos_constant": lambda value: f"const float eos_constant = {float(value)};\n",
    "delta_t": lambda value: f"const float delta_t = {float(value)};\n",
    "viscosity": lambda value: f"const float viscosity = {float(value)};\n",
    "cohesion": lambda value: f"const float cohesion = {float(value)};\n",
    "adhesion": lambda value: f"const float adhesion = {float(value)};\n",
    "offset": lambda value: f"const vec3 offset = vec3({float(value[0])}, {float(value[1])}, {float(value[2])});\n",
    "VOXEL_GROUP_SIZE": lambda value: f"const int VOXEL_GROUP_SIZE = {int(value)};\n" if abs(int(value)-value) < 1e-8 else print(f"VOXEL_GROUP_SIZE is an integer, while provided with another type, {type(value)}!"),
    "particle_volume": lambda value: f"const float particle_volume = {float(value)};\n",
    "Coeff_Poly6_2d": lambda value: f"const float Coeff_Poly6_2d = {float(value)};\n",
    "Coeff_Poly6_3d": lambda value: f"const float Coeff_Poly6_3d = {float(value)};\n",
    "Coeff_Spiky_2d": lambda value: f"const float Coeff_Spiky_2d = {float(value)};\n",
    "Coeff_Spiky_3d": lambda value: f"const float Coeff_Spiky_3d = {float(value)};\n",
    "Coeff_Viscosity_2d": lambda value: f"const float Coeff_Viscosity_2d = {float(value)};\n",
    "Coeff_Viscosity_3d": lambda value: f"const float Coeff_Viscosity_3d = {float(value)};\n",
    "Coeff_Wendland_3d": lambda value: f"const float Coeff_Wendland_3d = {float(value)};\n",
    "MAX_PARTICLE_MASS": lambda value: f"const float MAX_PARTICLE_MASS = {float(value)};\n",
    "ORIGINAL_PARTICLE_MASS": lambda value: f"const float ORIGINAL_PARTICLE_MASS = {float(value)};\n",
}


def complete_shader(value: dict):
    for src, export_file in zip(shader_src_list, shader_export_list):
        head = []
        const = [const_variables[k](value[k]) for k in value.keys()]
        body = []
        status = "head"  # current file read status
        with open(src, "r") as f:
            for row in f:
                if re.findall(find_const, row):
                    status = "body"
                    continue
                if status == "head":
                    head.append(row)
                elif status == "body":
                    body.append(row)
        with open(export_file, "w") as f:
            f.writelines(head)
            if status == "body":  # this shader has constants to be updated
                f.writelines(const)
                f.writelines(body)
            f.close()


if __name__ == "__main__":
    pi = 3.141592653589793
    value_dict = {
        "PI": 3.141592653589793,
        "n_boundary_particle": 10383,
        "n_voxel": 48235,
        "h": 0.01,
        "r": 0.0025,
        "voxel_memory_length": 2912,
        "voxel_block_size": 960,
        "rest_dense": 1000,
        "eos_constant": 32142.0,
        "delta_t": 0.00005,
        "viscosity": 0.0001,
        "cohesion": 0.0001,
        "adhesion": 0.0001,
        "offset": [-0.434871, -0.690556, -0.245941],
        "VOXEL_GROUP_SIZE": 300000,
        "particle_volume": 6.545e-08,
        "Coeff_Poly6_2d": 4 / (pi * 0.01**8),
        "Coeff_Poly6_3d": 315 / (64 * pi * 0.01**9),
        "Coeff_Spiky_2d": 10 / (pi * 0.01**5),
        "Coeff_Spiky_3d": 15 / (pi * 0.01**6),
        "Coeff_Viscosity_2d": 40 / (pi * 0.01**2),
        "Coeff_Viscosity_3d": 15 / (2 * pi * 0.01**3)
    }
    complete_shader(value_dict)

