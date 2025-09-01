#ifndef FFT_VISUALIZER_H
#define FFT_VISUALIZER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 
#include <cufft.h>

typedef struct {
    GLuint program;
    GLuint vao, vbo;
    cufftHandle plan;
    cudaGraphicsResource_t cuda_vbo_resource;
    int window_width;
    int window_height;
} VisualizerResources;

bool initVisualizer(VisualizerResources* pVisRes, int window_width, int window_height);

void processAndDraw(VisualizerResources* pVisRes, float* d_audioData, int bufferSize);

void cleanupVisualizer(VisualizerResources* pVisRes);

#endif
