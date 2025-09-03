#include "application.h"
#include <stdio.h>

application::application() : window_(nullptr), fft_visualizer_(WINDOW_WIDTH, WINDOW_HEIGHT) {}

application::~application() {
    shutdown();
}

bool application::initialize() {
    // Inicializar GLFW
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: Fallo al inicializar GLFW.\n");
        return false;
    }

    // Configurar el perfil de OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Crear la ventana de GLFW
    window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Audio Visualizer CUDA", NULL, NULL);
    if (!window_) {
        fprintf(stderr, "ERROR: Fallo al crear la ventana de GLFW.\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);

    // Inicializar GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "ERROR: Fallo al inicializar GLEW.\n");
        glfwTerminate();
        return false;
    }

    // Inicializar la captura de audio
    if (!audio_capture_.initialize()) {
        fprintf(stderr, "ERROR: Fallo al inicializar la captura de audio.\n");
        return false;
    }

    // Inicializar el visualizador
    if (!fft_visualizer_.initialize()) {
        fprintf(stderr, "ERROR: Fallo al inicializar el visualizador.\n");
        return false;
    }

    return true;
}

void application::run() {
    while (!glfwWindowShouldClose(window_)) {
        // Capturar y transferir los datos de audio a la GPU
        float* d_audio_data = audio_capture_.capture_and_transfer_audio();
        int audio_buffer_size = audio_capture_.get_buffer_size();

        if (d_audio_data && audio_buffer_size > 0) {
            // Actualizar el visualizador con los nuevos datos de audio
            fft_visualizer_.update(d_audio_data, audio_buffer_size);
            cudaFree(d_audio_data);
        }

        // Dibujar el visualizador en la pantalla
        fft_visualizer_.draw();

        glfwSwapBuffers(window_);
        glfwPollEvents();
    }
}

void application::shutdown() {
    // Limpiar recursos
    fft_visualizer_.cleanup();
    audio_capture_.cleanup();

    // Terminar GLFW
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}
