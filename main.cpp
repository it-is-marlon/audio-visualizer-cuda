#include <stdio.h>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "audio_capture.h"
#include "fft_visualizer.h"
#include <assert.h>

#define CUFFT_FFT_SIZE 512

void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "Error de GLFW: %s\n", description);
}

int main() {
    AudioResources audioRes = { NULL, NULL, NULL };
    VisualizerResources visRes = { 0, 0, 0, NULL, 0, 0, 0 };

    HRESULT hr = InitializeAudioCapture(&audioRes);
    if (FAILED(hr)) {
        printf("Error: Fallo al inicializar la captura de audio. Saliendo...\n");
        CleanupAudioCapture(&audioRes);
        return 1;
    }
    printf("WASAPI inicializado correctamente para captura en modo loopback.\n");

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        printf("Error: Fallo al inicializar GLFW.\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Audio Visualizer", NULL, NULL);
    if (!window) {
        printf("Error: Fallo al crear la ventana de GLFW.\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        printf("Error: Fallo al inicializar GLEW.\n");
        return -1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    if (!initVisualizer(&visRes, width, height)) {
        printf("Error: Fallo al inicializar el visualizador.\n");
        glfwTerminate();
        return -1;
    }

    while (!glfwWindowShouldClose(window)) {
        float* d_audioData = NULL;
        int audioBufferSize = 0;
        hr = CaptureAndTransferAudio(audioRes.pCaptureClient, &d_audioData, &audioBufferSize);

        if (SUCCEEDED(hr) && d_audioData && audioBufferSize > 0) {
            processAndDraw(&visRes, d_audioData, audioBufferSize);
            cudaDeviceSynchronize();
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (d_audioData) {
            cudaFree(d_audioData);
        }
    }

    cleanupVisualizer(&visRes);
    glfwTerminate();
    CleanupAudioCapture(&audioRes);

    return 0;
}
