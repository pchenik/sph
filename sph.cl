const float kernel_radius = 16;
const float particle_mass = 65;
const float poly6 = 315.f / (65.f * float(M_PI) * pow(kernel_radius, 9));
const float spiky_grad = -45.f / (float(M_PI) * pow(kernel_radius, 6));
const float visc_laplacian = 45.f / (float(M_PI) * pow(kernel_radius, 6));
const float gas_const = 2000.f;
const float rest_density = 1000.f;
const float visc_const = 250.f;
const float2 G = (0.f, 12000*-9.8f);


kernel void compute_density_and_pressure(global float* density,
                                        global float* pressure,
                                        global float2* positions) {
    const int glob_id = get_global_id(0);
    const int loc_id = get_local_id(0);

}

kernel void compute_forces(global float* density,
                           global float* pressure,
                           global float2* positions) {

    const int glob_id = get_global_id(0);

}

kernel void compute_positions(global float* density,
                              global float* pressure,
                              global float2* positions) {

    const int glob_id = get_global_id(0);


}