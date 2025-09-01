#include "fft_visualizer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
})";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main() {
   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
})";

__global__ void processAudioData(const float* d_input, cufftComplex* d_output, int num_samples) {
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

__global__ void updateVBO(cufftComplex* d_fft_buffer, float* d_vbo_data, int num_bars) {
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

bool initVisualizer(VisualizerResources* pVisRes, int window_width, int window_height) {
    if (!pVisRes) return false;

    pVisRes->window_width = window_width;
    pVisRes->window_height = window_height;

    // Crear VAO, VBO y el programa de shaders
    glGenVertexArrays(1, &pVisRes->vao);
    glGenBuffers(1, &pVisRes->vbo);

    glBindVertexArray(pVisRes->vao);
    glBindBuffer(GL_ARRAY_BUFFER, pVisRes->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 256 * 6 * 2, NULL, GL_DYNAMIC_DRAW);

    // Configurar atributos del vértice
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Desvincular VAO y VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Compilar y vincular shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    pVisRes->program = glCreateProgram();
    glAttachShader(pVisRes->program, vertexShader);
    glAttachShader(pVisRes->program, fragmentShader);
    glLinkProgram(pVisRes->program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&pVisRes->cuda_vbo_resource, pVisRes->vbo, cudaGraphicsMapFlagsNone);
    if (cudaStatus != cudaSuccess) {
        printf("Error: Fallo al registrar el VBO con CUDA: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    cufftResult cufftStatus = cufftPlan1d(&pVisRes->plan, 512, CUFFT_R2C, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        printf("Error: Fallo al crear el plan de cuFFT.\n");
        return false;
    }

    return true;
}

void processAndDraw(VisualizerResources* pVisRes, float* d_audioData, int bufferSize) {
    if (!pVisRes || !d_audioData) return;

    static cufftComplex* d_fft_buffer = NULL;
    if (!d_fft_buffer) {
        cudaMalloc(&d_fft_buffer, sizeof(cufftComplex) * 512);
    }

    int blockSize = 256;
    int gridSize = (bufferSize + blockSize - 1) / blockSize;
    processAudioData << <gridSize, blockSize >> > (d_audioData, d_fft_buffer, bufferSize);

    cufftExecR2C(pVisRes->plan, (cufftReal*)d_audioData, d_fft_buffer);

    float* d_vbo_data = NULL;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &pVisRes->cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_data, &num_bytes, pVisRes->cuda_vbo_resource);

    int numBars = 256;
    gridSize = (numBars + blockSize - 1) / blockSize;
    updateVBO << <gridSize, blockSize >> > (d_fft_buffer, d_vbo_data, numBars);

    cudaGraphicsUnmapResources(1, &pVisRes->cuda_vbo_resource, 0);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(pVisRes->program);
    glBindVertexArray(pVisRes->vao);
    glDrawArrays(GL_TRIANGLES, 0, numBars * 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

void cleanupVisualizer(VisualizerResources* pVisRes) {
    if (!pVisRes) return;
    cufftDestroy(pVisRes->plan);
    cudaGraphicsUnregisterResource(pVisRes->cuda_vbo_resource);
    glDeleteProgram(pVisRes->program);
    glDeleteBuffers(1, &pVisRes->vbo);
    glDeleteVertexArrays(1, &pVisRes->vao);
}
