#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <sstream>

#include <GL/glew.h>
#include <GL/glut.h>

#include "opengl.hh"
#include "vector.hh"

using clock_type = std::chrono::high_resolution_clock;
using float_duration = std::chrono::duration<float>;
using vec2 = Vector<float, 2>;

// Original code: https://github.com/cerrno/mueller-sph
constexpr const float kernel_radius = 16;
constexpr const float particle_mass = 65;
constexpr const float poly6 = 315.f / (65.f*float(M_PI)*std::pow(kernel_radius,9));
constexpr const float spiky_grad = -45.f / (float(M_PI)*std::pow(kernel_radius,6));
constexpr const float visc_laplacian = 45.f / (float(M_PI)*std::pow(kernel_radius,6));
constexpr const float gas_const = 2000.f;
constexpr const float rest_density = 1000.f;
constexpr const float visc_const = 250.f;
constexpr const vec2 G(0.f, 12000*-9.8f);

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
} opencl;

struct Particle {

    vec2 position;
    vec2 velocity;
    vec2 force;
    float density;
    float pressure;

    Particle() = default;
    inline explicit Particle(vec2 x): position(x) {}

};

std::vector<Particle> particles;
std::vector<float> particles_positions;

void generate_particles() {
    std::random_device dev;
    std::default_random_engine prng(dev());
    float jitter = 1;
    std::uniform_real_distribution<float> dist_x(-jitter,jitter);
    std::uniform_real_distribution<float> dist_y(-jitter,jitter);
    int ni = 15;
    int nj = 40;
    float x0 = window_width * 0.25f;
    float x1 = window_width * 0.75f;
    float y0 = window_height * 0.20f;
    float y1 = window_height * 1.00f;
    float step = 1.5f * kernel_radius;
    for (float x = x0; x < x1; x += step) {
        for (float y = y0; y < y1; y += step) {
            particles.emplace_back(vec2{x + dist_x(prng), y + dist_y(prng)});
        }
    }
    std::clog << "No. of particles: " << particles.size() << std::endl;
}

void compute_density_and_pressure() {
    const auto kernel_radius_squared = kernel_radius*kernel_radius;
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        float sum = 0;
        for (auto& b : particles) {
            auto sd = square(b.position-a.position);
            if (sd < kernel_radius_squared) {
                sum += particle_mass*poly6*std::pow(kernel_radius_squared-sd, 3);
            }
        }
        a.density = sum;
        a.pressure = gas_const*(a.density - rest_density);
    }
}

void compute_forces() {
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        vec2 pressure_force(0.f, 0.f);
        vec2 viscosity_force(0.f, 0.f);
        for (auto& b : particles) {
            if (&a == &b) { continue; }
            auto delta = b.position - a.position;
            auto r = length(delta);
            if (r < kernel_radius) {
                pressure_force += -unit(delta) * particle_mass * (a.pressure + b.pressure)
                    / (2.f * b.density)
                    * spiky_grad * std::pow(kernel_radius - r,2.f);
                viscosity_force += visc_const * particle_mass*(b.velocity - a.velocity)
                    / b.density * visc_laplacian * (kernel_radius - r);
            }
        }
        vec2 gravity_force = G * a.density;
        a.force = pressure_force + viscosity_force + gravity_force;
    }
}

void compute_positions() {
    const float time_step = 0.0008f;
    const float eps = kernel_radius;
    const float damping = -0.5f;
    #pragma omp parallel for
    for (auto& p : particles) {
        // forward Euler integration
        p.velocity += time_step * p.force / p.density;
        p.position += time_step * p.velocity;
        // enforce boundary conditions
        if (p.position(0) - eps < 0.0f) {
            p.velocity(0) *= damping;
            p.position(0) = eps;
        }
        if (p.position(0) + eps > window_width) {
            p.velocity(0) *= damping;
            p.position(0) = window_width-eps;
        }
        if (p.position(1) - eps < 0.0f) {
            p.velocity(1) *= damping;
            p.position(1) = eps;
        }
        if (p.position(1) + eps > window_height) {
            p.velocity(1) *= damping;
            p.position(1) = window_height-eps;
        }
    }
}

enum class Version { CPU, GPU };
Version version = Version::GPU;

void on_display() {
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,fbo); }
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluOrtho2D(0, window_width, 0, window_height);
    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    for (int i = 0; i < particles.size(); ++i) {
        switch(version) {
            case Version::CPU: glVertex2f(particles[i].position(0), particles[i].position(1)); break;
            case Version::GPU: glVertex2f(particles_positions[2 * i], particles_positions[2 * i + 1]); break;
            default: break;
        }
    }
    glEnd();
    glutSwapBuffers();
    if (no_screen) { glReadBuffer(GL_RENDERBUFFER); }
    recorder.record_frame();
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,0); }
}

void on_idle_cpu() {
    if (particles.empty()) { generate_particles(); }
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    auto t0 = clock_type::now();
    compute_density_and_pressure();
    compute_forces();
    compute_positions();
    auto t1 = clock_type::now();
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}

struct Buffers {
    Buffers () = default;

    void init () {

        /*std::cout << "BEFORE_EVERYTHING " << particles.size() << "\n";

        int buf_size = particles.size()*sizeof(cl_float);

        cl::Buffer temp(opencl.context, CL_MEM_READ_WRITE, buf_size);

        std::cout << "KEK1" << "\n";*/

        density = cl::Buffer(opencl.context, CL_MEM_READ_WRITE, particles.size() * sizeof(float));

        std::cout << "KEK2" << "\n";

        pressure = {opencl.context, CL_MEM_READ_WRITE, particles.size() * sizeof(float)};
        forces = {opencl.context, CL_MEM_READ_WRITE, 2 * particles.size() * sizeof(float)};
        velocities = {opencl.context, CL_MEM_READ_WRITE, 2 * particles.size() * sizeof(float)};


        std::cout << "BEFORE" << "\n";

        particles_positions.resize(2 * particles.size());
        std::iota(particles_positions.begin(), particles_positions.end(), 0);

        for(const auto& x: particles_positions)
            std::cout << x << " ";
        std::cout << "\n" << "\n";

        std::for_each(particles_positions.begin(), particles_positions.end(), [](float &coord){
            coord = particles[int(coord) / 2].position(int(coord) & 1);
        });

        positions =  {opencl.queue, begin(particles_positions), end(particles_positions), true};

        std::cout << "AFTER" << "\n";

    }

    cl::Buffer density;
    cl::Buffer pressure;
    cl::Buffer forces;
    cl::Buffer velocities;
    cl::Buffer positions;
} cl_buffers;

void compute_density_and_pressure_gpu() {
    cl::Event ev_kernel;
    cl::Kernel kernel(opencl.program, "compute_density_and_pressure");
    kernel.setArg(0, cl_buffers.density);
    kernel.setArg(1, cl_buffers.pressure);
    kernel.setArg(2, cl_buffers.positions);
    //4 warp per 1 multiprocessor
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(particles.size()), cl::NDRange(128),
                                      NULL, &ev_kernel);
    ev_kernel.wait();
    opencl.queue.flush();
}

void compute_forces_gpu() {
    cl::Event ev_kernel;
    cl::Kernel kernel(opencl.program, "compute_forces");
    kernel.setArg(0, cl_buffers.density);
    kernel.setArg(1, cl_buffers.pressure);
    kernel.setArg(2, cl_buffers.forces);
    kernel.setArg(3, cl_buffers.velocities);
    kernel.setArg(4, cl_buffers.positions);
    //4 warp per 1 multiprocessor
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(particles.size()), cl::NDRange(128),
                                      NULL, &ev_kernel);
    ev_kernel.wait();
    opencl.queue.flush();
}

void compute_positions_gpu() {
    cl::Event ev_kernel;
    cl::Kernel kernel(opencl.program, "compute_positions");
    kernel.setArg(0, cl_buffers.density);
    kernel.setArg(1, cl_buffers.forces);
    kernel.setArg(2, cl_buffers.velocities);
    kernel.setArg(3, cl_buffers.positions);
    //4 warp per 1 multiprocessor
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(particles.size()), cl::NDRange(128),
                                      NULL, &ev_kernel);
    ev_kernel.wait();
    opencl.queue.flush();
}

void on_idle_gpu() {
    //std::clog << "GPU version is not implemented!" << std::endl; std::exit(1);
    if (particles.empty()) {
        generate_particles();
        std::cout << "HERE1" << "\n";
        cl_buffers.init();
        std::cout << "HERE2" << "\n";
    }
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    auto t0 = clock_type::now();
    compute_density_and_pressure_gpu();
    compute_forces_gpu();
    compute_positions_gpu();
    auto t1 = clock_type::now();
    //copy the positions back on cpu
    cl::copy(opencl.queue, cl_buffers.positions, begin(particles_positions), end(particles_positions));
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}

void on_keyboard(unsigned char c, int x, int y) {
    switch(c) {
        case ' ':
            generate_particles();
            break;
        case 'r':
        case 'R':
            particles.clear();
            generate_particles();
            break;
    }
}

void print_column_names() {
    std::clog << std::setw(20) << "Frame duration";
    std::clog << std::setw(20) << "Frames per second";
    std::clog << '\n';
}

void openCL_init() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return;
        }
        cl::Platform platform = platforms[0];
        std::clog << platforms.size() << " Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
                { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';

        std::ifstream ifs("../sph.cl");
        std::string src( (std::istreambuf_iterator<char>(ifs) ),
                         (std::istreambuf_iterator<char>()    ) );

        cl::Program program(context, src);

        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }

        cl::CommandQueue queue(context, device);
        opencl = {platform, device, context, program, queue};

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
                  << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return;
    }
}

int main(int argc, char* argv[]) {
    //enum class Version { CPU, GPU };
    //Version version = Version::CPU;
    /*if (argc == 2) {
        std::string str(argv[1]);
        for (auto& ch : str) { ch = std::tolower(ch); }
        if (str == "gpu") { version = Version::GPU; }
    }*/
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(window_width, window_height);
	glutInit(&argc, argv);
	glutCreateWindow("SPH");
	glutDisplayFunc(on_display);
    glutReshapeFunc(on_reshape);
    switch (version) {
        case Version::CPU: glutIdleFunc(on_idle_cpu); break;
        case Version::GPU: {
            openCL_init();
            glutIdleFunc(on_idle_gpu);
            break;
        }
        default: return 1;
    }
	glutKeyboardFunc(on_keyboard);
    glewInit();
	init_opengl(kernel_radius);
    print_column_names();
	glutMainLoop();
    return 0;
}
