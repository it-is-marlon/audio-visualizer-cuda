#include "fft_visualizer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

const char* vertex_shader_source = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
})";

const char* fragment_shader_source = R"(
#version 330 core
out vec4 FragColor;
void main() {
   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
})";

__global__ void process_audio_data(const float* d_input, cufftComplex* d_output, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        d_output[idx].x = d_input[idx];
        d_output[idx].y = 0.0f;
    }
    else {
        d_output[idx].x = 0.0f;
        d_output[idx].y = 0.0f;
    }
}

__global__ void update_vbo(cufftComplex* d_fft_buffer, float* d_vbo_data, int num_bars) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bars) {
        float magnitude = sqrtf(d_fft_buffer[idx].x * d_fft_buffer[idx].x + d_fft_buffer[idx].y * d_fft_buffer[idx].y);

        // Usar una escala logarítmica para una mejor visualización de la magnitud
        float bar_height = fminf(log10f(magnitude + 1.0f) * 0.1f, 1.0f);
        float bar_width = 2.0f / num_bars;
        float x_pos = (float)idx * bar_width - 1.0f;

        // Triángulo 1
        d_vbo_data[idx * 12 + 0] = x_pos;
        d_vbo_data[idx * 12 + 1] = -1.0f;
        d_vbo_data[idx * 12 + 2] = x_pos + bar_width;
        d_vbo_data[idx * 12 + 3] = -1.0f;
        d_vbo_data[idx * 12 + 4] = x_pos;
        d_vbo_data[idx * 12 + 5] = bar_height - 1.0f;

        // Triángulo 2
        d_vbo_data[idx * 12 + 6] = x_pos + bar_width;
        d_vbo_data[idx * 12 + 7] = -1.0f;
        d_vbo_data[idx * 12 + 8] = x_pos + bar_width;
        d_vbo_data[idx * 12 + 9] = bar_height - 1.0f;
        d_vbo_data[idx * 12 + 10] = x_pos;
        d_vbo_data[idx * 12 + 11] = bar_height - 1.0f;
    }
}

fft_visualizer::fft_visualizer(int width, int height)
    : window_width_(width), window_height_(height), program_(0), vao_(0), vbo_(0), plan_(0), cuda_vbo_resource_(nullptr), d_audio_data_old_(nullptr), d_fft_buffer_(nullptr) {
}

fft_visualizer::~fft_visualizer() {
    cleanup();
}

bool fft_visualizer::initialize() {
    // Crear VAO, VBO y el programa de shaders
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 256 * 6 * 2, NULL, GL_DYNAMIC_DRAW);

    // Configurar atributos del vértice
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Desvincular VAO y VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Compilar y vincular shaders
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);

    program_ = glCreateProgram();
    glAttachShader(program_, vertex_shader);
    glAttachShader(program_, fragment_shader);
    glLinkProgram(program_);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_, vbo_, cudaGraphicsMapFlagsNone);
    if (cudaStatus != cudaSuccess) {
        printf("Error: Fallo al registrar el VBO con CUDA: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    cufftResult cufftStatus = cufftPlan1d(&plan_, 512, CUFFT_R2C, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        printf("Error: Fallo al crear el plan de cuFFT.\n");
        return false;
    }

    return true;
}

void fft_visualizer::update(float* d_audio_data, int buffer_size) {
    if (!d_fft_buffer_) {
        cudaMalloc(&d_fft_buffer_, sizeof(cufftComplex) * 512);
    }

    int blockSize = 256;
    int gridSize = (buffer_size + blockSize - 1) / blockSize;
    process_audio_data << <gridSize, blockSize >> > (d_audio_data, d_fft_buffer_, buffer_size);

    cufftExecR2C(plan_, (cufftReal*)d_audio_data, d_fft_buffer_);

    float* d_vbo_data = NULL;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource_, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_data, &num_bytes, cuda_vbo_resource_);

    int num_bars = 256;
    gridSize = (num_bars + blockSize - 1) / blockSize;
    update_vbo << <gridSize, blockSize >> > (d_fft_buffer_, d_vbo_data, num_bars);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource_, 0);
}

void fft_visualizer::draw() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 256 * 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

void fft_visualizer::cleanup() {
    if (d_fft_buffer_) cudaFree(d_fft_buffer_);
    if (plan_) cufftDestroy(plan_);
    if (cuda_vbo_resource_) cudaGraphicsUnregisterResource(cuda_vbo_resource_);
    if (program_) glDeleteProgram(program_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
}
