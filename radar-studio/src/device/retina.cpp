#include "retina.h"

#include <common/os/network.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <sstream>
#include <thread>

#define CHECK_READ(exp) \
	do { \
		if (!(exp)) \
		{ \
			m_log << "[error] [parse] failed to read data\n"; \
			return false; \
		} \
	} while (0)

namespace retina
{
	enum
	{
		kHttpPort = 8000,
		kDevicePort = 29172,
		kProveTimeoutMs = 100
	};

	enum : uint32_t
	{
		kPacketMagic = 0xABCD4321,
	};
	
	enum : uint64_t
	{
		kFrameHeaderSize = 8,
		kMaxFrameSize = 16 * 1024 * 1024,
		kFrameMagic = 0x0807060504030201,
		kTargetOffset = 48056
	};

	struct RetinaPacketHeaderRaw
	{
		uint32_t _reserved0;
		uint32_t magic;
		uint32_t _reserved1;
		uint32_t _reserved2;
		uint32_t packageSize;
		uint32_t _reserved3;
		uint32_t _reserved4;
		uint32_t _reserved5;
		uint32_t _reserved6;
	};

	struct RetinaFrameHeaderRaw
	{
		uint64_t magic;
		uint32_t frameCount;
		uint32_t targetNumber;
	};

	struct RetinaTargetRaw
	{
		float x;
		float y;
		TargetStatus status;
		uint32_t targetId;
		float reserved0;
		float reserved1;
		float reserved2;
	};

	namespace detail
	{
		uint64_t get_steady_us()
		{
			auto now = std::chrono::steady_clock::now();
			return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
		}

		uint32_t parseIpv4(const std::string& ip)
		{
			asio::error_code ec;
			const auto address = asio::ip::make_address_v4(ip, ec);
			return ec ? 0u : address.to_uint();
		}

		class ByteReader
		{
		public:
			ByteReader(const uint8_t* data, size_t size) :
				m_data(data),
				m_size(size),
				m_offset(0)
			{
			}
		
			size_t remaining() const
			{
				return m_size - m_offset;
			}
		
			size_t offset() const
			{
				return m_offset;
			}
		
			bool seek(size_t offset)
			{
				if (offset > m_size)
					return false;

				m_offset = offset;

				return true;
			}
		
			template <typename T>
			bool read(T& out)
			{
				if (m_offset + sizeof(T) > m_size)
					return false;
		
				std::memcpy(&out, m_data + m_offset, sizeof(T));
		
				m_offset += sizeof(T);
		
				return true;
			}
		
			template <typename T>
			bool peek(T& out)
			{
				if (m_offset + sizeof(T) > m_size)
					return false;
		
				std::memcpy(&out, m_data + m_offset, sizeof(T));
		
				return true;
			}
		
		private:
			const uint8_t* m_data;
			size_t m_size;
			size_t m_offset;
		};
	}

	/////////////////////////////////////////////////////////////////////////////////////////////// Device Finder Implementation

	DeviceFinder::DeviceFinder(std::ostream& log) :
		m_log(log)
	{
	}

	DeviceFinder::Result DeviceFinder::find(const std::string& local_ip, const std::string& subnet_mask, std::string& out_host)
	{
		assert(m_ioContexts.empty());

		uint32_t network = 0;
		uint32_t first = 0;
		uint32_t last = 0;

		if (!buildScanRange(local_ip, subnet_mask, network, first, last))
			return Result::Failed;

		m_log << "[info] local ip: " << local_ip << ", mask: " << subnet_mask << '\n';
		m_log << "[info] scanning range: " << formatAddress(first) << " - " << formatAddress(last) << '\n';

		const uint32_t total_threads = std::thread::hardware_concurrency();
		const uint32_t worker_count = std::clamp(total_threads == 0 ? 8u : total_threads, 1u, 32u);
		std::atomic<uint32_t> next_ip(first);
		std::atomic<bool> found(false);
		std::mutex result_mutex;
		std::vector<std::thread> workers;

		workers.reserve(worker_count);

		m_canceled.store(false, std::memory_order_release);
		m_ioContexts.resize(worker_count, nullptr);

		for (uint32_t i = 0; i < worker_count; ++i)
		{
			workers.emplace_back([&, this, first, last](uint32_t worker_id)
				{
					asio::io_context io_ctx;

					{
						std::lock_guard<std::mutex> lock(m_mutex);
						m_ioContexts[worker_id] = &io_ctx;
					}

					while (!found.load(std::memory_order_acquire) && !m_canceled.load(std::memory_order_acquire))
					{
						const uint32_t current = next_ip.fetch_add(1, std::memory_order_relaxed);
						if (current < first || current > last)
							break;

						const std::string ip = formatAddress(current);
						if (!probeHost(io_ctx, ip))
							continue;

						bool expected = false;
						if (!found.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
							break;

						std::lock_guard<std::mutex> lock(result_mutex);
						out_host = ip;
					}

					{
						std::lock_guard<std::mutex> lock(m_mutex);
						m_ioContexts[worker_id] = nullptr;
					}
				}, i);
		}

		for (std::thread& worker : workers)
			worker.join();
		m_ioContexts.clear();

		if (m_canceled.load(std::memory_order_acquire))
		{
			m_log << "[info] device discovery canceled\n";
			return Result::Canceled;
		}

		if (out_host.empty())
		{
			m_log << "[error] retina: no retina device found on local subnet\n";
			return Result::Failed;
		}

		m_log << "[info] found candidate device at " << out_host << '\n';
		return Result::Success;
	}

	void DeviceFinder::cancel()
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		m_canceled.store(true, std::memory_order_release);

		for (asio::io_context* io_ctx : m_ioContexts)
			if (io_ctx) io_ctx->stop();
	}

	bool DeviceFinder::buildScanRange(const std::string& local_ip, const std::string& subnet_mask, uint32_t& out_network, uint32_t& out_first, uint32_t& out_last)
	{
		const uint32_t ip = detail::parseIpv4(local_ip);
		const uint32_t mask = detail::parseIpv4(subnet_mask);

		if (ip == 0 || mask == 0)
			return false;

		out_network = ip & mask;

		const uint32_t broadcast = out_network | ~mask;
		out_first = out_network + 1;
		out_last = broadcast > 0 ? broadcast - 1 : broadcast;

		if (out_first > out_last)
			return false;

		return true;
	}

	bool DeviceFinder::probeHost(asio::io_context& io, const std::string& ip)
	{
		auto timeout = std::chrono::milliseconds(kProveTimeoutMs);

		if (!probeTcpPort(io, ip, kDevicePort, timeout))
			return false;
		if (!probeHttpPort(io, ip, timeout))
			return false;
		return true;
	}

	bool DeviceFinder::probeTcpPort(asio::io_context& io, const std::string& ip, uint16_t port, std::chrono::milliseconds timeout)
	{
		asio::error_code ec;
		tcp::endpoint endpoint(asio::ip::make_address(ip, ec), port);

		if (ec) return false;

		tcp::socket socket(io);
		asio::steady_timer timer(io);

		bool connected = false;
		bool completed = false;

		timer.expires_after(timeout);
		timer.async_wait([&](const std::error_code& timer_ec)
		{
			if (!timer_ec && !completed)
				socket.cancel();
		});

		socket.async_connect(endpoint, [&](const std::error_code& connect_ec)
		{
			completed = true;
			connected = !connect_ec;
			timer.cancel();
		});

		io.restart();
		io.run();
		socket.close();

		return connected;
	}

	bool DeviceFinder::probeHttpPort(asio::io_context& io, const std::string& ip, std::chrono::milliseconds timeout)
	{
		asio::error_code ec;
		tcp::endpoint endpoint(asio::ip::make_address(ip, ec), kHttpPort);

		if (ec) return false;

		tcp::socket socket(io);
		asio::steady_timer timer(io);
		std::array<char, 256> response {};

		bool saw_http = false;
		bool completed = false;

		timer.expires_after(timeout);
		timer.async_wait([&](const std::error_code& timer_ec)
		{
			if (!timer_ec && !completed)
				socket.cancel();
		});

		socket.async_connect(endpoint, [&](const std::error_code& connect_ec)
		{
			if (connect_ec)
			{
				completed = true;
				timer.cancel();
				return;
			}

			static const std::string kRequest = "GET / HTTP/1.1\r\nHost: radar\r\nConnection: close\r\n\r\n";
			asio::async_write(socket, asio::buffer(kRequest), [&](const std::error_code& write_ec, std::size_t)
			{
				if (write_ec)
				{
					completed = true;
					timer.cancel();
					return;
				}

				socket.async_read_some(asio::buffer(response), [&](const std::error_code& read_ec, std::size_t bytes)
				{
					completed = true;
					timer.cancel();
					if (read_ec && read_ec != asio::error::eof)
						return;
					const std::string_view view(response.data(), bytes);
					saw_http = view.find("HTTP/") != std::string_view::npos;
				});
			});
		});

		io.restart();
		io.run();
		socket.close();

		return saw_http;
	}

	std::string DeviceFinder::formatAddress(uint32_t address) const
	{
		return asio::ip::address_v4(address).to_string();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////// Device Client

	DeviceClient::DeviceClient(asio::io_context& io, const std::string& host, std::ostream& log_stream) :
		m_io(io),
		m_log(log_stream),
		m_resolver(io),
		m_socket(io),
		m_frameCountLimit(kFrameCountLimitDefault)
	{
		m_resolver.async_resolve(
			host,
			std::to_string(kDevicePort),
			[this, host](const std::error_code& ec, tcp::resolver::results_type endpoints)
			{
				m_log << "[info] try resolving " << host << ':' << kDevicePort << "...\n";
	
				if (ec)
				{
					m_log << "[error] retina: resolve failed: " << ec.message() << '\n';
					return;
				}

				connectAsync(endpoints);
			});
	}

	DeviceClient::~DeviceClient()
	{
		shutdown();
	}

	void DeviceClient::setOnConnected(std::function<void()> callback)
	{
		m_onConnected = std::move(callback);
	}

	void DeviceClient::setOnDisconnected(std::function<void()> callback)
	{
		m_onDisconnected = std::move(callback);
	}

	void DeviceClient::setOnFrame(std::function<void(const Frame&)> callback)
	{
		m_onFrame = std::move(callback);
	}

	void DeviceClient::setFrameCountLimit(uint32_t limit)
	{
		m_frameCountLimit = limit;
	}

	void DeviceClient::getFrames(std::function<void(const std::deque<Frame>&)> callback) const
	{
		if (callback)
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			callback(m_frames);
		}
	}

	double DeviceClient::getFrameRate() const
	{
		if (m_totalDurationUs > 0)
			return (m_totalFrames - 1) * 1e6 / m_totalDurationUs;
		return 0.0;
	}

	double DeviceClient::getBandwidthMbps(uint64_t interval_ms) const
	{
		uint64_t now = detail::get_steady_us();

		while (!m_bandwidthSamples.empty() && now - m_bandwidthSamples.front().first > interval_ms * 1000) {
			m_bandwidthTotalBytes -= m_bandwidthSamples.front().second;
			m_bandwidthSamples.pop_front();
		}

		return (m_bandwidthTotalBytes * 8.0) / 1000000.0;
	}

	const DeviceInfo& DeviceClient::getDeviceInfo() const
	{
		return m_deviceInfo;
	}

	void DeviceClient::shutdown()
	{
		std::error_code ec;
		m_resolver.cancel();
		m_socket.cancel(ec);
		m_socket.shutdown(tcp::socket::shutdown_both, ec);
		m_socket.close(ec);
	}
	
	void DeviceClient::connectAsync(tcp::resolver::results_type endpoints)
	{
		asio::async_connect(
			m_socket,
			endpoints,
			[this](const std::error_code& ec, const tcp::endpoint& endpoint)
			{
				if (ec)
				{
					m_log << "[error] retina: connect failed: " << ec.message() << '\n';
					return;
				}

				asio::socket_base::keep_alive option(true);
				m_socket.set_option(option);
	
				std::string mac;
				std::string ip = endpoint.address().to_string();
				std::string port = std::to_string(endpoint.port());

				m_log << "[info] connected to "
					<< endpoint.address().to_string()
					<< ':' << endpoint.port() << '\n';

				m_deviceInfo.ip = ip;
				m_deviceInfo.model = "RETINA-4SN";
				if (os::getMacAddress(ip, mac))
				{
					m_deviceInfo.mac = mac;
					m_log << "[info] device MAC address: " << mac << '\n';
				}
				else
				{
					m_log << "[warning] failed to get device MAC address\n";
					m_deviceInfo.mac = "unknown";
				}

				if (m_onConnected)
					m_onConnected();

				m_readBuf.resize(8192);

				doRead();
			});
	}

	void DeviceClient::doRead()
	{
		m_socket.async_read_some(
			asio::buffer(m_readBuf),
			[this](const std::error_code& ec, std::size_t bytes)
			{
				if (ec)
				{
					m_log << "[error] retina: read failed: " << ec.message() << '\n';
					if (m_onDisconnected)
						m_onDisconnected();
					return;
				}
	
				auto first = m_readBuf.begin();
				auto last = m_readBuf.begin() + bytes;
	
				m_streamBuf.insert(m_streamBuf.end(), first, last);

				m_bandwidthSamples.emplace_back(detail::get_steady_us(), bytes);
				m_bandwidthTotalBytes += bytes;
	
				while (true)
				{
					if (!tryExtractPacket(m_streamBuf, m_packetBuf))
						break;
	
					Frame frame;
					if (parseSinglePacket(m_packetBuf, frame))
					{
						{
							std::lock_guard<std::mutex> lock(m_mutex);

							while (m_frames.size() >= m_frameCountLimit)
							{
								m_totalDurationUs -= m_frames.front().deltaUs;
								m_frames.pop_front();
							}

							m_totalDurationUs += frame.deltaUs;
							m_totalFrames = m_frames.size();
						}

						if (m_onFrame)
							m_onFrame(frame);

						m_frames.push_back(std::move(frame));
					}
					else
					{
						m_log << "[error] retina: failed to parse frame\n";
					}
				}
	
				doRead();
			});
	}
	
	bool DeviceClient::tryExtractPacket(std::vector<uint8_t>& stream_buf, std::vector<uint8_t>& out_packet_buf)
	{
		size_t packet_magic_offset;
	
		if (!findPacketMagic(stream_buf, packet_magic_offset))
		{
			// Keep last few bytes in case magic spans chunk boundary
			if (stream_buf.size() > 7)
				stream_buf.erase(stream_buf.begin(), stream_buf.end() - 7);
	
			return false;
		}
	
		if (packet_magic_offset < 4)
		{
			return false;
		}
	
		if (packet_magic_offset > 4)
		{
			stream_buf.erase(stream_buf.begin(), stream_buf.begin() + packet_magic_offset - 4);
		}
	
		if (stream_buf.size() < sizeof(RetinaPacketHeaderRaw))
			return false;
	
		RetinaPacketHeaderRaw packet_header{};
		memcpy(&packet_header, stream_buf.data(), sizeof(RetinaPacketHeaderRaw));
	
		assert(packet_header.magic == kPacketMagic);
	
		size_t packet_size = sizeof(RetinaPacketHeaderRaw) + packet_header.packageSize;
	
		if (stream_buf.size() >= packet_size)
		{
			out_packet_buf.assign(stream_buf.begin(), stream_buf.begin() + packet_size);
			stream_buf.erase(stream_buf.begin(), stream_buf.begin() + packet_size);
	
			return true;
		}
	
		return false;
	}
	
	bool DeviceClient::parseSinglePacket(const std::vector<uint8_t>& packet_buf, Frame& out_frame)
	{
		uint64_t now = detail::get_steady_us();
		
		if (m_lastFrameTimepoint == 0)
			out_frame.deltaUs = 0;
		else
			out_frame.deltaUs = now - m_lastFrameTimepoint;
		m_lastFrameTimepoint = now;

		detail::ByteReader reader(packet_buf.data(), packet_buf.size());
	
		RetinaPacketHeaderRaw packet_header{};
		RetinaFrameHeaderRaw frame_header{};
	
		CHECK_READ(reader.read(packet_header));
	
		assert(packet_header.magic == kPacketMagic);
		assert(packet_header.packageSize == packet_buf.size() - sizeof(RetinaPacketHeaderRaw));
	
		CHECK_READ(reader.read(frame_header));
	
		if (frame_header.magic != kFrameMagic)
		{
			m_log << "[error] retina: invalid frame magic: " << std::hex << frame_header.magic << std::dec << '\n';
			return false;
		}
	
		uint32_t point_number = frame_header.targetNumber;
		uint32_t target_number = 0;
	
		out_frame.packetSize = packet_header.packageSize;
		out_frame.frameCount = frame_header.frameCount;
	
		out_frame.points.reserve(point_number);
		for (size_t i = 0; i < point_number; ++i)
		{
			Point p{};
	
			CHECK_READ(reader.read(p.x));
			CHECK_READ(reader.read(p.y));
			CHECK_READ(reader.read(p.z));
			CHECK_READ(reader.read(p.doppler));
			CHECK_READ(reader.read(p.power));
			p.targetId = -1;
	
			out_frame.points.push_back(p);
		}

		for (uint32_t i = 0; i < point_number; ++i)
			CHECK_READ(reader.read(out_frame.points[i].targetId));

		if (reader.remaining() == 0)
			return true;
	
		CHECK_READ(reader.peek(frame_header));

		// Retina devices place the target section at a fixed offset; the emulator writes it right after the points.
		if (frame_header.magic != kFrameMagic)
			reader.seek(kTargetOffset);

		CHECK_READ(reader.read(frame_header));
	
		if (frame_header.magic != kFrameMagic)
		{
			m_log << "[error] retina: invalid frame magic for target section: " << std::hex << frame_header.magic << std::dec << '\n';
			return false;
		}
	
		assert(frame_header.frameCount == out_frame.frameCount);
	
		target_number = frame_header.targetNumber;
		out_frame.targets.reserve(target_number);
	
		for (uint32_t i = 0; i < target_number; ++i)
		{
			RetinaTargetRaw target_raw;
			CHECK_READ(reader.read(target_raw));
	
			Target target{};
			target.minx = target.miny = target.minz = std::numeric_limits<float>::max();
			target.maxx = target.maxy = target.maxz = std::numeric_limits<float>::lowest();
	
			target.x = target_raw.x;
			target.y = target_raw.y;
			target.status = target_raw.status;
			target.targetId = target_raw.targetId;
	
			for (const auto& p : out_frame.points)
			{
				if (p.targetId == target.targetId)
				{
					target.minx = std::min(target.minx, p.x);
					target.maxx = std::max(target.maxx, p.x);
					target.miny = std::min(target.miny, p.y);
					target.maxy = std::max(target.maxy, p.y);
					target.minz = std::min(target.minz, p.z);
					target.maxz = std::max(target.maxz, p.z);
				}
			}
	
			out_frame.targets.push_back(target);
		}
	
		return true;
	}
	
	bool DeviceClient::findPacketMagic(const std::vector<uint8_t>& buffer, size_t& out_offset)
	{
		if (buffer.size() < 4)
			return false;
	
		for (size_t i = 0; i + 4 <= buffer.size(); ++i)
		{
			if (*reinterpret_cast<const uint32_t*>(buffer.data() + i) == kPacketMagic) {
				out_offset = i;
				return true;
			}
		}
	
		return false;
	}
}
