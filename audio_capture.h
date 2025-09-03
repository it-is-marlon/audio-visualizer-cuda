#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <mmdeviceapi.h>
#include <audioclient.h>
#include <cuda_runtime.h>

class audio_capture {
public:
    audio_capture();
    ~audio_capture();

    bool initialize();
    float* capture_and_transfer_audio();
    void cleanup();
    int get_buffer_size();

private:
    IMMDevice* p_device_;
    IAudioClient* p_audio_client_;
    IAudioCaptureClient* p_capture_client_;
    int buffer_size_;
};

#endif
