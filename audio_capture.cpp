#include "audio_capture.h"
#include <stdio.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <Functiondiscoverykeys_devpkey.h>
#include <cuda_runtime.h>
#include <comdef.h>

#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MS   10000

HRESULT InitializeAudioCapture(AudioResources* pRes) {
    if (!pRes) return E_POINTER;
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) return hr;

    IMMDeviceEnumerator* pEnumerator = NULL;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&pEnumerator);
    if (FAILED(hr)) {
        CoUninitialize();
        return hr;
    }

    hr = pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &pRes->pDevice);
    if (FAILED(hr)) {
        pEnumerator->Release();
        CoUninitialize();
        return hr;
    }
    pEnumerator->Release();

    hr = pRes->pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pRes->pAudioClient);
    if (FAILED(hr)) {
        CoUninitialize();
        return hr;
    }

    WAVEFORMATEX* pwfx = NULL;
    hr = pRes->pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        CoUninitialize();
        return hr;
    }

    hr = pRes->pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, REFTIMES_PER_SEC, 0, pwfx, NULL);
    if (FAILED(hr)) {
        CoTaskMemFree(pwfx);
        CoUninitialize();
        return hr;
    }

    hr = pRes->pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pRes->pCaptureClient);
    if (FAILED(hr)) {
        CoTaskMemFree(pwfx);
        CoUninitialize();
        return hr;
    }

    hr = pRes->pAudioClient->Start();
    if (FAILED(hr)) {
        CoTaskMemFree(pwfx);
        CoUninitialize();
        return hr;
    }

    CoTaskMemFree(pwfx);
    return S_OK;
}

HRESULT CaptureAndTransferAudio(IAudioCaptureClient* pCaptureClient, float** d_audioBuffer, int* bufferSize) {
    if (!pCaptureClient || !d_audioBuffer || !bufferSize) return E_POINTER;

    UINT32 packetLength = 0;
    BYTE* pData = NULL;
    UINT32 numFramesInPacket = 0;
    DWORD flags = 0;

    HRESULT hr = pCaptureClient->GetNextPacketSize(&packetLength);
    if (FAILED(hr) || packetLength == 0) return S_OK;

    hr = pCaptureClient->GetBuffer(&pData, &numFramesInPacket, &flags, NULL, NULL);
    if (FAILED(hr) || numFramesInPacket == 0) return S_OK;

    float* hostAudioData = new float[numFramesInPacket];
    for (UINT32 i = 0; i < numFramesInPacket; ++i) {
        hostAudioData[i] = ((float*)pData)[i];
    }
    pCaptureClient->ReleaseBuffer(numFramesInPacket);

    *bufferSize = numFramesInPacket;
    cudaMalloc((void**)d_audioBuffer, sizeof(float) * numFramesInPacket);
    cudaMemcpy(*d_audioBuffer, hostAudioData, sizeof(float) * numFramesInPacket, cudaMemcpyHostToDevice);

    delete[] hostAudioData;
    return S_OK;
}

void CleanupAudioCapture(AudioResources* pRes) {
    if (!pRes) return;
    if (pRes->pCaptureClient) pRes->pCaptureClient->Release();
    if (pRes->pAudioClient) pRes->pAudioClient->Stop();
    if (pRes->pAudioClient) pRes->pAudioClient->Release();
    if (pRes->pDevice) pRes->pDevice->Release();
    CoUninitialize();
}
