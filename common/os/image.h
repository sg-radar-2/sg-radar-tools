#pragma once

#include <cstdint>
#include <vector>

namespace os
{
	struct ImageRGBA
	{
		int width = 0;
		int height = 0;
		std::vector<std::uint8_t> pixels; // width * height * 4, straight alpha
	};

	bool loadImageRGBA(const char* path, ImageRGBA& out_image);
}
