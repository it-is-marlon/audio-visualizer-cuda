#ifndef APPLICATION_H
#define APPLICATION_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "audio_capture.h"
#include "fft_visualizer.h"

class application {
public:
    application();
    ~application();

    bool initialize();
    void run();
    void shutdown();

private:
    GLFWwindow* window_;
    audio_capture audio_capture_;
    fft_visualizer fft_visualizer_;

    const int WINDOW_WIDTH = 1280;
    const int WINDOW_HEIGHT = 720;
};

#endif
