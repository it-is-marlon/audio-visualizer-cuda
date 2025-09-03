#ifndef FFT_VISUALIZER_H
#define FFT_VISUALIZER_H

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> 
#include <cufft.h>

class fft_visualizer {
public:
    fft_visualizer(int width, int height);
    ~fft_visualizer();

    bool initialize();
    void update(float* d_audio_data, int buffer_size);
    void draw();
    void cleanup();

private:
    int window_width_;
    int window_height_;
    GLuint program_;
    GLuint vao_, vbo_;
    cufftHandle plan_;
    cudaGraphicsResource_t cuda_vbo_resource_;
    float* d_audio_data_old_;
    cufftComplex* d_fft_buffer_;

    // Helper functions for shaders
    GLuint create_shader(GLenum type, const char* source);
    GLuint create_program(GLuint vertex_shader, GLuint fragment_shader);
};

#endif
