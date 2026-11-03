#include "image.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

#include <string>

namespace os
{
	bool loadImageRGBA(const char* path, ImageRGBA& out_image)
	{
		out_image = ImageRGBA{};
		if (path == nullptr || path[0] == '\0')
			return false;

#ifdef _WIN32
		const int wide_len = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
		if (wide_len <= 0)
			return false;
		std::wstring wide_path(static_cast<size_t>(wide_len), L'\0');
		MultiByteToWideChar(CP_UTF8, 0, path, -1, wide_path.data(), wide_len);

		const HRESULT init_hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
		const bool need_uninit = SUCCEEDED(init_hr);

		bool ok = false;
		IWICImagingFactory* factory = nullptr;
		IWICBitmapDecoder* decoder = nullptr;
		IWICBitmapFrameDecode* frame = nullptr;
		IWICFormatConverter* converter = nullptr;

		do
		{
			if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))))
				break;
			if (FAILED(factory->CreateDecoderFromFilename(wide_path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder)))
				break;
			if (FAILED(decoder->GetFrame(0, &frame)))
				break;
			if (FAILED(factory->CreateFormatConverter(&converter)))
				break;
			if (FAILED(converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
				break;

			UINT width = 0, height = 0;
			if (FAILED(converter->GetSize(&width, &height)) || width == 0 || height == 0)
				break;

			const UINT stride = width * 4;
			out_image.pixels.resize(static_cast<size_t>(stride) * height);
			if (FAILED(converter->CopyPixels(nullptr, stride, static_cast<UINT>(out_image.pixels.size()), out_image.pixels.data())))
				break;

			out_image.width = static_cast<int>(width);
			out_image.height = static_cast<int>(height);
			ok = true;
		} while (false);

		if (converter) converter->Release();
		if (frame) frame->Release();
		if (decoder) decoder->Release();
		if (factory) factory->Release();
		if (need_uninit)
			CoUninitialize();

		if (!ok)
			out_image = ImageRGBA{};
		return ok;
#else
		return false;
#endif
	}
}
