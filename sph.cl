#define kernel_radius  16
#define eps 16
#define kernel_radius_squared (float)(kernel_radius*kernel_radius)
#define particle_mass  65
#define poly6  315.f/(65.f*(float)(M_PI)*pow((float)kernel_radius,(float)9))
#define spiky_grad  -45.f/((float)(M_PI)*pow((float)kernel_radius,(float)6))
#define visc_laplacian  45.f/((float)(M_PI)*pow((float)kernel_radius,(float)6))
#define gas_const  2000.f
#define rest_density  1000.f
#define visc_const  250.f
#define G (float2)(0.f, 12000*-9.8f)
//#define window_width 800.f
//#define window_height 600.f

/*#define kernel_radius 16.f
#define particle_mass 65.f
#define gas_const 2000.f
#define rest_density 1000.f
#define visc_const 250.f
#define G (float2)(0.f, 12000*-9.8f)
#define window_width 800.f
#define window_height 600.f
#define eps kernel_radius
//#define damping -0.5f
#define poly6 315.f / (65.f * (float)(M_PI) * pow(kernel_radius, 9.f))
#define spiky_grad -45.f / ((float)(M_PI) * pow(kernel_radius, 6.f))
#define visc_laplacian 45.f / ((float)(M_PI) * pow(kernel_radius, 6.f))*/

/*float2 unit(float2 vector) {
    float length = sqrt((vector.x * vector.x) + (vector.y * vector.y));
    return vector / length;
}*/

/*void upper_bounce(float* pos, float* velocity, float limit) {
    if (*pos + eps > limit) {
        *velocity *= damping;
        *pos = limit - eps;
    }
}

void lower_bounce(float* pos, float* velocity) {
    if (*pos - eps < 0.0f) {
        *velocity *= damping;
        *pos = eps;
    }
}*/

kernel void compute_density_and_pressure(global float* density,
                                         global float* pressure,
                                         global float2* positions) {
    const int glob_id = get_global_id(0);
    const int N = get_global_size(0);
    float square;
    float sum = 0;
    //float kernel_radius_squared = kernel_radius * kernel_radius;
    float2 delta;
    float2 cur_pos = positions[glob_id];

    for (int i = 0; i < N; ++i) {
            delta = cur_pos - positions[i];
            square = (delta.x * delta.x) + (delta.y * delta.y);
            if (square < kernel_radius_squared) {
                sum += particle_mass * poly6 * pow(kernel_radius_squared - square, 3);
            }
        }

    density[glob_id] = sum;
    pressure[glob_id] = gas_const * (sum - rest_density);
}

kernel void compute_forces(global float* density,
                           global float* pressure,
                           global float2* forces,
                           global float2* velocities,
                           global float2* positions
                           ) {

    const int glob_id = get_global_id(0);
    const int N = get_global_size(0);
    float r;
    float2 pressure_force = 0;
    float2 viscosity_force = 0;
    float2 delta;
    float2 cur_pos = positions[glob_id];
    float2 cur_velocity = velocities[glob_id];

    float cur_pressure = pressure[glob_id];
    float cur_density = density[glob_id];

    for (int i = 0; i < N; ++i)
        if (i != glob_id) {
            delta = positions[i] - cur_pos;
            r = sqrt((delta.x * delta.x) + (delta.y * delta.y));
            //r = length(delta);
            if (r < kernel_radius) {
                pressure_force += -normalize(delta) * particle_mass * (cur_pressure + pressure[i]) / (2.f * density[i]) * spiky_grad * pow(kernel_radius - r, 2.f);
                viscosity_force += visc_const * particle_mass * (velocities[i] - cur_velocity) / density[i] * visc_laplacian * (kernel_radius - r);
            }
        }

    float2 gravity_force = G * cur_density;
    forces[glob_id] = pressure_force + viscosity_force + gravity_force;
}

kernel void compute_positions(global float* density,
                              global float2* forces,
                              global float2* velocities,
                              global float2* positions,
                              int window_width, int window_height) {

    const int glob_id = get_global_id(0);
    const float time_step = 0.0008f;
    const float damping = -0.5f;

    velocities[glob_id] += time_step * forces[glob_id] / density[glob_id];
    positions[glob_id] += time_step * velocities[glob_id];

    float2 cur_pos = positions[glob_id];
    float2 cur_velocity = velocities[glob_id];

    // enforce boundary conditions

    if (cur_pos.x - eps < 0.0f) {
        cur_velocity.x *= damping;
        cur_pos.x = eps;
    }
    if (cur_pos.x + eps > window_width) {
        cur_velocity.x *= damping;
        cur_pos.x = window_width - eps;
    }
    if (cur_pos.y - eps < 0.0f) {
        cur_velocity.y *= damping;
        cur_pos.y = eps;
    }
    if (cur_pos.y + eps > window_height) {
        cur_velocity.y *= damping;
        cur_pos.y = window_height - eps;
    }

    /*lower_bounce(&cur_pos, &cur_velocity);
    upper_bounce(&cur_pos, &cur_velocity, window_width);

    lower_bounce(&cur_pos + sizeof(float), &cur_velocity + sizeof(float));
    upper_bounce(&cur_pos + sizeof(float), &cur_velocity + sizeof(float), window_height);*/

    velocities[glob_id] = cur_velocity;
    positions[glob_id] = cur_pos;
}