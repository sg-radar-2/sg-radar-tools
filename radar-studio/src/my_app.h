#pragma once

#include <common/os/network.h>
#include <common/pcr/pcr_type.h>
#include <common/util/debug_console.h>

#include <atomic>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct GLFWwindow;
struct ImGuiContext;
struct ImGuiSettingsHandler;
struct ImGuiTextBuffer;

class AnalysisWindow;
class ConsoleWindow;
class DockingWindow;
class PointCloudWindow;
class RecordingWindow;
class SensorWindow;

namespace asio
{
	class io_context;
}

namespace retina
{
	struct DeviceInfo;
	class DeviceClient;
	class DeviceFinder;
}

struct WindowRect
{
	int x = -1;
	int y = -1;
	int width = -1;
	int height = -1;

	bool hasPos() const { return x != -1 && y != -1; }
};

class MyApp
{
public:
	enum class ConnectionStatus
	{
		Disconnected,
		Finding,
		Connecting,
		Connected,
		Failed
	};

	enum class Pictogram
	{
		Standing,
		Sitting,
		Lying,
		Walking,
		Count
	};

	static MyApp& get();

	MyApp();
	~MyApp();

	void run();

	void connect(const std::string& host = {});
	void cancelOrDisconnect();

	ConnectionStatus getConnectionStatus() const;
	bool getDeviceInfoUpdated() const;

	const os::LocalNetworkInfo& getLocalNetworkInfo() const;
	const retina::DeviceInfo& getDeviceInfo() const;
	const pcr::Frame& getLastFrame() const;
	float getLastBandwidthMbps() const;
	float getLastFrameRate() const;

	void setSensorSpec(const pcr::SensorSpec& spec);
	void resetSensorSpec();

	// OpenGL texture name (0 if the image failed to load). White glyph with alpha, so it can be tinted.
	unsigned int getPictogramTexture(Pictogram pictogram) const;

private:
	void initWindow();
	void setupWindows();
	void loadPictograms();
	void releasePictograms();
	void setupDockLayout(unsigned int dockspace_id);

	void updateWindows();
	void updateSensor();
	void drawFrame();
	void drawMenu();
	void drawFooter(float height);
	void drawConnectionStatus();
	void limitFrameRate(float fps) const;

	void connectToDeviceAsync(const std::string& host);

	std::ostream& log();

private:
	static void glfwErrorCallback(int error, const char* description);
	static void glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos);
	static void glfwWindowSizeCallback(GLFWwindow* window, int width, int height);
	static void glfwWindowIconifyCallback(GLFWwindow* window, int iconified);
	static void glfwWindowMaximizeCallback(GLFWwindow* window, int maximized);

	static void* windowReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);
	static void windowReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line);
	static void windowWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf);

	static void* sensorSpecReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);
	static void sensorSpecReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line);
	static void sensorSpecWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf);

	void setTheme(int theme);
	static void* themeReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);
	static void themeReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line);
	static void themeWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf);

private:
	static MyApp* s_instance;

	GLFWwindow* m_window = nullptr;
	WindowRect m_windowRestored;
	WindowRect m_windowCurrent;
	bool m_windowMinimized = false;
	bool m_windowMaximized = false;
	bool m_dockLayoutInitialized = false;
	bool m_hasSavedDockLayout = false;
	int m_theme = 0;

	DebugConsole m_console;
	std::vector<std::unique_ptr<DockingWindow>> m_allWindows;
	std::vector<PointCloudWindow*> m_pointWindows;
	ConsoleWindow* m_consoleWindow = nullptr;
	SensorWindow* m_sensorWindow = nullptr;
	RecordingWindow* m_recordingWindow = nullptr;
	AnalysisWindow* m_analysisWindow = nullptr;
	DockingWindow* m_activeWindow = nullptr;

	unsigned int m_pictogramTextures[static_cast<int>(Pictogram::Count)] = {};

	os::LocalNetworkInfo m_networkInfo;
	float m_networkRefreshTimer = 0.5f;
	std::mutex m_networkMutex;

	std::unique_ptr<retina::DeviceFinder> m_deviceFinder;
	std::mutex m_finderMutex;

	std::unique_ptr<retina::DeviceClient> m_client;
	std::mutex m_clientMutex;
	std::atomic<ConnectionStatus> m_connectionStatus{ ConnectionStatus::Disconnected };
	std::atomic<bool> m_connectionUpdated{ false };
	bool m_deviceInfoUpdated = false;
	bool m_pendingSensorSpecNotify = false;

	std::unique_ptr<asio::io_context> m_io;
	std::thread m_ioThread;

	std::unique_ptr<retina::DeviceInfo> m_deviceInfo;
	pcr::Frame m_lastFrame;
	float m_lastBandwidthMbps = 0.0f;
	float m_lastFrameRate = 0.0f;
};
