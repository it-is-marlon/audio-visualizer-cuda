#include "audio_capture.h"
#include <stdio.h>
#include <Functiondiscoverykeys_devpkey.h>
#include <comdef.h>

#define REFTIMES_PER_SEC 10000000
#define REFTIMES_PER_MS 10000

audio_capture::audio_capture()
    : p_device_(nullptr), p_audio_client_(nullptr), p_capture_client_(nullptr), buffer_size_(0) {
}

audio_capture::~audio_capture() {
    cleanup();
}

bool audio_capture::initialize() {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al inicializar COM: %s\n", _com_error(hr).ErrorMessage());
        return false;
    }

    IMMDeviceEnumerator* p_enumerator = NULL;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&p_enumerator);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al crear IMMDeviceEnumerator: %s\n", _com_error(hr).ErrorMessage());
        CoUninitialize();
        return false;
    }

    hr = p_enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &p_device_);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al obtener el endpoint de audio predeterminado: %s\n", _com_error(hr).ErrorMessage());
        p_enumerator->Release();
        CoUninitialize();
        return false;
    }
    p_enumerator->Release();

    hr = p_device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&p_audio_client_);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al activar IAudioClient: %s\n", _com_error(hr).ErrorMessage());
        return false;
    }

    WAVEFORMATEX* pwfx = NULL;
    hr = p_audio_client_->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al obtener el formato de mezcla: %s\n", _com_error(hr).ErrorMessage());
        return false;
    }

    hr = p_audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, REFTIMES_PER_SEC, 0, pwfx, NULL);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al inicializar IAudioClient: %s\n", _com_error(hr).ErrorMessage());
        CoTaskMemFree(pwfx);
        return false;
    }

    hr = p_audio_client_->GetService(__uuidof(IAudioCaptureClient), (void**)&p_capture_client_);
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al obtener IAudioCaptureClient: %s\n", _com_error(hr).ErrorMessage());
        CoTaskMemFree(pwfx);
        return false;
    }

    hr = p_audio_client_->Start();
    if (FAILED(hr)) {
        fprintf(stderr, "Error: Fallo al iniciar la captura: %s\n", _com_error(hr).ErrorMessage());
        CoTaskMemFree(pwfx);
        return false;
    }

    CoTaskMemFree(pwfx);
    return true;
}

float* audio_capture::capture_and_transfer_audio() {
    UINT32 packet_length = 0;
    BYTE* p_data = NULL;
    UINT32 num_frames_in_packet = 0;
    DWORD flags = 0;

    HRESULT hr = p_capture_client_->GetNextPacketSize(&packet_length);
    if (FAILED(hr) || packet_length == 0) return nullptr;

    hr = p_capture_client_->GetBuffer(&p_data, &num_frames_in_packet, &flags, NULL, NULL);
    if (FAILED(hr) || num_frames_in_packet == 0) return nullptr;

    float* d_audio_buffer = nullptr;
    cudaMalloc((void**)&d_audio_buffer, sizeof(float) * num_frames_in_packet);
    cudaMemcpy(d_audio_buffer, p_data, sizeof(float) * num_frames_in_packet, cudaMemcpyHostToDevice);

    p_capture_client_->ReleaseBuffer(num_frames_in_packet);

    buffer_size_ = num_frames_in_packet;
    return d_audio_buffer;
}

void audio_capture::cleanup() {
    if (p_capture_client_) p_capture_client_->Release();
    if (p_audio_client_) p_audio_client_->Stop();
    if (p_audio_client_) p_audio_client_->Release();
    if (p_device_) p_device_->Release();
    CoUninitialize();
}

int audio_capture::get_buffer_size() {
    return buffer_size_;
}
