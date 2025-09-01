#include <mmdeviceapi.h>
#include <audioclient.h>
#include <stdio.h>

typedef struct {
    IMMDevice* pDevice;
    IAudioClient* pAudioClient;
    IAudioCaptureClient* pCaptureClient;
} AudioResources;

HRESULT InitializeAudioCapture(AudioResources* pRes);

HRESULT CaptureAndTransferAudio(IAudioCaptureClient* pCaptureClient, float** d_audioBuffer, int* bufferSize);

void CleanupAudioCapture(AudioResources* pRes);
