#pragma once

#include "retina.h"

#include <memory>
#include <mutex>
#include <ostream>

namespace retina
{
	// Relays the targets of every received frame to another program on the same PC
	// (tools/behavior/situation_viewer.pyw live mode) over UDP. If nobody is listening the
	// datagrams are simply dropped, so this never affects Radar Studio itself.
	//
	// Datagram (little-endian, one frame per datagram):
	//   char[4]    magic "SGLV"
	//   uint32     version (= 1)
	//   uint32     frameCount
	//   uint64     deltaUs
	//   SensorSpec sensorSpec (64 bytes, same as pcr_format.h)
	//   uint32     targetCount
	//   target[targetCount]:
	//     Target   (40 bytes, same as pcr_format.h: x, y, status, targetId, minx..maxz)
	//     uint32   pointCount      (points assigned to this target in this frame)
	//     float    dopplerMean, dopplerAbsMean, dopplerStd
	class LiveRelay
	{
	public:
		static constexpr uint16_t kDefaultPort = 29190;
		static constexpr uint32_t kVersion = 1;

		explicit LiveRelay(std::ostream& log, uint16_t port = kDefaultPort);
		~LiveRelay();

		void setSensorSpec(const SensorSpec& spec);
		void send(const Frame& frame);

	private:
		std::ostream& m_log;
		asio::io_context m_io;
		asio::ip::udp::socket m_socket;
		asio::ip::udp::endpoint m_endpoint;
		std::mutex m_mutex;
		SensorSpec m_spec{};
		std::vector<uint8_t> m_buf;
		bool m_ok = false;
		bool m_loggedError = false;
	};
}
