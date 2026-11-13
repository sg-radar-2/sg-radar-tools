#include "live_relay.h"

#include <cmath>
#include <cstring>

namespace retina
{
	namespace
	{
		template <typename T>
		void put(std::vector<uint8_t>& buf, const T& v)
		{
			const auto* p = reinterpret_cast<const uint8_t*>(&v);
			buf.insert(buf.end(), p, p + sizeof(T));
		}
	}

	LiveRelay::LiveRelay(std::ostream& log, uint16_t port) :
		m_log(log),
		m_socket(m_io),
		m_endpoint(asio::ip::make_address_v4("127.0.0.1"), port)
	{
		std::error_code ec;
		m_socket.open(asio::ip::udp::v4(), ec);
		if (ec)
		{
			m_log << "[warning] live relay: UDP socket open failed: " << ec.message() << '\n';
			return;
		}
		m_ok = true;
		m_log << "[info] live relay: sending targets to udp://127.0.0.1:" << port
			<< " (tools/behavior/situation_viewer.pyw live mode)\n";
	}

	LiveRelay::~LiveRelay()
	{
		std::error_code ec;
		m_socket.close(ec);
	}

	void LiveRelay::setSensorSpec(const SensorSpec& spec)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_spec = spec;
	}

	void LiveRelay::send(const Frame& frame)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_ok)
			return;

		m_buf.clear();
		m_buf.reserve(96 + frame.targets.size() * 56);
		m_buf.insert(m_buf.end(), { 'S', 'G', 'L', 'V' });
		put(m_buf, kVersion);
		put(m_buf, frame.frameCount);
		put(m_buf, frame.deltaUs);
		put(m_buf, m_spec);
		put(m_buf, static_cast<uint32_t>(frame.targets.size()));

		for (const Target& t : frame.targets)
		{
			uint32_t n = 0;
			double sum = 0.0, abs_sum = 0.0, sq_sum = 0.0;
			for (const Point& p : frame.points)
			{
				if (p.targetId < 0 || static_cast<uint32_t>(p.targetId) != t.targetId)
					continue;
				++n;
				sum += p.doppler;
				abs_sum += std::fabs(p.doppler);
				sq_sum += static_cast<double>(p.doppler) * p.doppler;
			}
			const float mean = n ? static_cast<float>(sum / n) : NAN;
			const float abs_mean = n ? static_cast<float>(abs_sum / n) : NAN;
			const double var = n ? sq_sum / n - (sum / n) * (sum / n) : 0.0;
			const float stdev = n ? static_cast<float>(std::sqrt(var > 0.0 ? var : 0.0)) : NAN;

			put(m_buf, t);
			put(m_buf, n);
			put(m_buf, mean);
			put(m_buf, abs_mean);
			put(m_buf, stdev);
		}

		std::error_code ec;
		m_socket.send_to(asio::buffer(m_buf), m_endpoint, 0, ec);
		if (ec && !m_loggedError)
		{
			// Windows may report an error for an earlier datagram when nobody listens. Log once and keep sending.
			m_log << "[warning] live relay: send failed (ignored): " << ec.message() << '\n';
			m_loggedError = true;
		}
	}
}
