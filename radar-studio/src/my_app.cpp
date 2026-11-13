#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")

#include "my_app.h"

#include "device/live_relay.h"
#include "device/retina.h"
#include "ui/implot3d_custom.h"
#include "ui/window/analysis_window.h"
#include "ui/window/point_cloud_window.h"
#include "ui/window/recording_window.h"
#include "ui/window/sensor_window.h"

#include <common/os/image.h>
#include <common/ui/imgui_custom.h>
#include <common/ui/window/console_window.h>
#include <common/ui/window/docking_window.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <implot.h>
#include <implot3d.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>

#define FONT_AWESOME_PATH "../../asset/font/fa-solid-900.ttf"
#define PICTOGRAM_DIR "../../asset/img/"
#define CONSOLAS_FONT_PATH "C:/Windows/Fonts/consola.ttf"

namespace
{
	constexpr int kPointWindowCount = 4;
}

MyApp* MyApp::s_instance = nullptr;

MyApp& MyApp::get()
{
	assert(s_instance != nullptr);
	return *s_instance;
}

MyApp::MyApp()
{
	s_instance = this;
	m_io = std::make_unique<asio::io_context>();
	m_deviceInfo = std::make_unique<retina::DeviceInfo>();

	initWindow();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	if (io.IniFilename != nullptr)
		m_hasSavedDockLayout = std::filesystem::exists(io.IniFilename);

	setupWindows();
	loadPictograms();

	MyGui::applyTheme(MyGui::Theme::Dark);
	MyPlot3D::applyTheme();

	ImGui::LoadIniSettingsFromDisk("imgui.ini");
	MyGui::applyTheme(static_cast<MyGui::Theme>(m_theme));

	if (m_sensorWindow)
		m_sensorWindow->syncDraftFromDevice();
	m_pendingSensorSpecNotify = true;

	m_liveRelay = std::make_unique<retina::LiveRelay>(log());
	m_liveRelay->setSensorSpec(m_deviceInfo->sensorSpec);   // sensor spec loaded from imgui.ini

	if (m_windowRestored.hasPos())
	{
		glfwSetWindowPos(m_window, m_windowRestored.x, m_windowRestored.y);
		glfwSetWindowSize(m_window, m_windowRestored.width, m_windowRestored.height);
		if (m_windowMinimized)
			glfwIconifyWindow(m_window);
		if (m_windowMaximized)
			glfwMaximizeWindow(m_window);
	}

	glfwShowWindow(m_window);
}

MyApp::~MyApp()
{
	cancelOrDisconnect();

	ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);

	releasePictograms();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot3D::DestroyContext();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_window);
	glfwTerminate();

	s_instance = nullptr;
}

void MyApp::run()
{
	while (!glfwWindowShouldClose(m_window))
	{
		glfwPollEvents();

		if (m_windowMinimized)
		{
			glfwWaitEventsTimeout(0.01);
			continue;
		}

		updateWindows();
		drawFrame();

		limitFrameRate(60.0f);
	}
}

void MyApp::connect(const std::string& host)
{
	if (m_connectionStatus.load() == ConnectionStatus::Connected)
		return;

	cancelOrDisconnect();

	if (!host.empty())
	{
		m_ioThread = std::thread([this, host]()
			{
				connectToDeviceAsync(host);
			});
		return;
	}

	m_connectionStatus.store(ConnectionStatus::Finding);

	m_ioThread = std::thread([this]()
		{
			os::LocalNetworkInfo network_info;
			std::string found_host;

			{
				std::lock_guard<std::mutex> lock(m_networkMutex);
				network_info = m_networkInfo;
			}
			{
				std::lock_guard<std::mutex> lock(m_finderMutex);
				m_deviceFinder = std::make_unique<retina::DeviceFinder>(log());
			}

			const auto res = m_deviceFinder->find(network_info.address, network_info.subnetMask, found_host);

			{
				std::lock_guard<std::mutex> lock(m_finderMutex);
				m_deviceFinder.reset();
			}

			if (res == retina::DeviceFinder::Result::Failed)
			{
				m_connectionStatus.store(ConnectionStatus::Failed);
				log() << "[error] network: failed to discover retina device on the local network\n";
				return;
			}
			if (res == retina::DeviceFinder::Result::Canceled)
			{
				m_connectionStatus.store(ConnectionStatus::Disconnected);
				log() << "[info] network: device discovery canceled\n";
				return;
			}

			log() << "[info] network: retina device found at " << found_host << '\n';
			connectToDeviceAsync(found_host);
		});
}

void MyApp::cancelOrDisconnect()
{
	if (m_connectionStatus.load() == ConnectionStatus::Finding)
	{
		{
			std::lock_guard<std::mutex> lock(m_finderMutex);
			if (m_deviceFinder)
				m_deviceFinder->cancel();
		}

		if (m_ioThread.joinable())
			m_ioThread.join();

		{
			std::lock_guard<std::mutex> lock(m_finderMutex);
			m_deviceFinder.reset();
		}
	}
	else if (m_connectionStatus.load() == ConnectionStatus::Connected)
	{
		{
			std::lock_guard<std::mutex> lock(m_clientMutex);
			if (m_client)
				m_client->shutdown();
		}

		m_io->stop();

		if (m_ioThread.joinable())
			m_ioThread.join();

		{
			std::lock_guard<std::mutex> lock(m_clientMutex);
			m_client.reset();
		}
	}
	else if (m_ioThread.joinable())
	{
		m_ioThread.join();
	}

	m_connectionStatus.store(ConnectionStatus::Disconnected);
	const retina::SensorSpec applied_spec = m_deviceInfo->sensorSpec;
	*m_deviceInfo = retina::DeviceInfo{};
	m_deviceInfo->sensorSpec = applied_spec;
}

MyApp::ConnectionStatus MyApp::getConnectionStatus() const
{
	return m_connectionStatus.load();
}

bool MyApp::getDeviceInfoUpdated() const
{
	return m_deviceInfoUpdated;
}

const os::LocalNetworkInfo& MyApp::getLocalNetworkInfo() const
{
	return m_networkInfo;
}

const retina::DeviceInfo& MyApp::getDeviceInfo() const
{
	return *m_deviceInfo;
}

const pcr::Frame& MyApp::getLastFrame() const
{
	return m_lastFrame;
}

float MyApp::getLastBandwidthMbps() const
{
	return m_lastBandwidthMbps;
}

float MyApp::getLastFrameRate() const
{
	return m_lastFrameRate;
}

void MyApp::setSensorSpec(const pcr::SensorSpec& spec)
{
	m_deviceInfo->sensorSpec = spec;
	m_pendingSensorSpecNotify = true;
	if (m_liveRelay)
		m_liveRelay->setSensorSpec(spec);
	if (ImGui::GetCurrentContext() != nullptr)
		ImGui::MarkIniSettingsDirty();
}

void MyApp::resetSensorSpec()
{
	setSensorSpec(pcr::SensorSpec{});
}

unsigned int MyApp::getPictogramTexture(Pictogram pictogram) const
{
	const int index = static_cast<int>(pictogram);
	if (index < 0 || index >= static_cast<int>(Pictogram::Count))
		return 0;
	return m_pictogramTextures[index];
}

void MyApp::loadPictograms()
{
	static const char* kFiles[] = {
		PICTOGRAM_DIR "stand.png",
		PICTOGRAM_DIR "sit.png",
		PICTOGRAM_DIR "lying.png",
		PICTOGRAM_DIR "walk.png",
	};
	static_assert(std::size(kFiles) == static_cast<size_t>(Pictogram::Count), "pictogram file list mismatch");

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

	for (int i = 0; i < static_cast<int>(Pictogram::Count); ++i)
	{
		os::ImageRGBA image;
		if (!os::loadImageRGBA(kFiles[i], image))
		{
			log() << "[error] failed to load pictogram: " << kFiles[i] << '\n';
			continue;
		}

		// Keep only the alpha mask so the glyph can be tinted with any color.
		for (size_t p = 0; p < image.pixels.size(); p += 4)
			image.pixels[p + 0] = image.pixels[p + 1] = image.pixels[p + 2] = 255;

		GLuint texture = 0;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Build the mip chain on the CPU (2x2 box filter); the icons are drawn far smaller than 512px.
		int width = image.width;
		int height = image.height;
		std::vector<std::uint8_t> level = std::move(image.pixels);
		for (int mip = 0;; ++mip)
		{
			glTexImage2D(GL_TEXTURE_2D, mip, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, level.data());
			if (width == 1 && height == 1)
				break;

			const int next_width = std::max(1, width / 2);
			const int next_height = std::max(1, height / 2);
			std::vector<std::uint8_t> next(static_cast<size_t>(next_width) * next_height * 4);
			for (int y = 0; y < next_height; ++y)
			{
				for (int x = 0; x < next_width; ++x)
				{
					const int x0 = std::min(x * 2, width - 1), x1 = std::min(x * 2 + 1, width - 1);
					const int y0 = std::min(y * 2, height - 1), y1 = std::min(y * 2 + 1, height - 1);
					for (int c = 0; c < 4; ++c)
					{
						const int sum =
							level[(static_cast<size_t>(y0) * width + x0) * 4 + c] +
							level[(static_cast<size_t>(y0) * width + x1) * 4 + c] +
							level[(static_cast<size_t>(y1) * width + x0) * 4 + c] +
							level[(static_cast<size_t>(y1) * width + x1) * 4 + c];
						next[(static_cast<size_t>(y) * next_width + x) * 4 + c] = static_cast<std::uint8_t>((sum + 2) / 4);
					}
				}
			}

			level = std::move(next);
			width = next_width;
			height = next_height;
		}

		glBindTexture(GL_TEXTURE_2D, 0);
		m_pictogramTextures[i] = texture;
	}
}

void MyApp::releasePictograms()
{
	for (unsigned int& texture : m_pictogramTextures)
	{
		if (texture != 0)
		{
			const GLuint name = texture;
			glDeleteTextures(1, &name);
			texture = 0;
		}
	}
}

void MyApp::initWindow()
{
	glfwSetErrorCallback(glfwErrorCallback);

	if (!glfwInit())
		throw std::runtime_error("[glfw] failed to initialize GLFW");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImPlot3D::CreateContext();

	ImGuiSettingsHandler ini_handler;
	ini_handler.TypeName = "GLFW_Window";
	ini_handler.TypeHash = ImHashStr("GLFW_Window");
	ini_handler.ReadOpenFn = windowReadOpen;
	ini_handler.ReadLineFn = windowReadLine;
	ini_handler.WriteAllFn = windowWriteAll;
	ini_handler.UserData = this;
	ImGui::GetCurrentContext()->SettingsHandlers.push_back(ini_handler);

	ImGuiSettingsHandler spec_handler;
	spec_handler.TypeName = "SensorSpec";
	spec_handler.TypeHash = ImHashStr("SensorSpec");
	spec_handler.ReadOpenFn = sensorSpecReadOpen;
	spec_handler.ReadLineFn = sensorSpecReadLine;
	spec_handler.WriteAllFn = sensorSpecWriteAll;
	spec_handler.UserData = this;
	ImGui::GetCurrentContext()->SettingsHandlers.push_back(spec_handler);

	ImGuiSettingsHandler theme_handler;
	theme_handler.TypeName = "App";
	theme_handler.TypeHash = ImHashStr("App");
	theme_handler.ReadOpenFn = themeReadOpen;
	theme_handler.ReadLineFn = themeReadLine;
	theme_handler.WriteAllFn = themeWriteAll;
	theme_handler.UserData = this;
	ImGui::GetCurrentContext()->SettingsHandlers.push_back(theme_handler);

	m_window = glfwCreateWindow(1280, 800, "Radar Studio", nullptr, nullptr);
	if (m_window == nullptr)
		throw std::runtime_error("[glfw] failed to create window");

	glfwSetWindowUserPointer(m_window, this);
	glfwSetWindowPosCallback(m_window, glfwWindowPosCallback);
	glfwSetWindowSizeCallback(m_window, glfwWindowSizeCallback);
	glfwSetWindowIconifyCallback(m_window, glfwWindowIconifyCallback);
	glfwSetWindowMaximizeCallback(m_window, glfwWindowMaximizeCallback);

	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1);

	if (!ImGui_ImplGlfw_InitForOpenGL(m_window, true))
		throw std::runtime_error("[imgui] failed to initialize GLFW backend");
	if (!ImGui_ImplOpenGL3_Init("#version 130"))
		throw std::runtime_error("[imgui] failed to initialize OpenGL3 backend");
}

void MyApp::setupWindows()
{
	auto& io = ImGui::GetIO();

	m_allWindows.emplace_back(m_consoleWindow = new ConsoleWindow(m_console));
	m_allWindows.emplace_back(m_sensorWindow = new SensorWindow());
	m_allWindows.emplace_back(m_recordingWindow = new RecordingWindow());

	if (ImFont* font = io.Fonts->AddFontFromFileTTF(CONSOLAS_FONT_PATH, 18.0f))
	{
		io.FontDefault = font;
		MyGui::mergeFont(FONT_AWESOME_PATH, 18.0f);
	}

	for (int i = 0; i < kPointWindowCount; ++i)
	{
		auto point_window = std::make_unique<PointCloudWindow>("Window" + std::to_string(i + 1));
		point_window->setRecordingWindow(m_recordingWindow);

		m_pointWindows.push_back(point_window.get());
		m_allWindows.push_back(std::move(point_window));
	}

	m_allWindows.emplace_back(m_analysisWindow = new AnalysisWindow());
	m_analysisWindow->setSourceRecordingWindow(m_recordingWindow);
	m_analysisWindow->setSourcePointWindows(m_pointWindows);

	for (auto& wnd : m_allWindows)
		wnd->initialize(&m_console.getOutput());
}

void MyApp::setupDockLayout(unsigned int dockspace_id)
{
	ImGui::DockBuilderRemoveNode(dockspace_id);
	ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
	ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);

	ImGuiID dock_main = dockspace_id;
	ImGuiID dock_right = 0;
	ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Right, 0.28f, &dock_right, &dock_main);

	ImGui::DockBuilderDockWindow("Point Cloud", dock_main);
	ImGui::DockBuilderDockWindow("Inspector", dock_right);
	ImGui::DockBuilderDockWindow("Console", dock_right);
	ImGui::DockBuilderDockWindow("Recording", dock_right);
	ImGui::DockBuilderDockWindow("XY Projection", dock_main);
	ImGui::DockBuilderDockWindow("XZ Projection", dock_main);
	ImGui::DockBuilderDockWindow("YZ Projection", dock_main);

	ImGui::DockBuilderFinish(dockspace_id);
}

void MyApp::updateWindows()
{
	const bool spec_notify = m_pendingSensorSpecNotify;
	updateSensor();

	if (spec_notify)
	{
		const auto& spec = m_deviceInfo->sensorSpec;
		for (auto& point_window : m_pointWindows)
			point_window->setSensorSpec(spec);
	}

	for (auto& wnd : m_allWindows)
		wnd->update();
}

void MyApp::updateSensor()
{
	const float dt = ImGui::GetIO().DeltaTime;

	m_networkRefreshTimer += dt;
	m_deviceInfoUpdated = m_pendingSensorSpecNotify;
	m_pendingSensorSpecNotify = false;

	if (m_networkRefreshTimer >= 1.0f)
	{
		os::LocalNetworkInfo new_info;
		if (os::getLocalNetworkInfo(new_info))
		{
			std::lock_guard lock(m_networkMutex);
			m_networkInfo = new_info;
		}
		else
		{
			log() << "[error] Failed to get local network info" << std::endl;
		}
		m_networkRefreshTimer = 0.0f;
	}

	if (m_connectionUpdated.load())
	{
		const retina::SensorSpec applied_spec = m_deviceInfo->sensorSpec;
		*m_deviceInfo = m_client ? m_client->getDeviceInfo() : retina::DeviceInfo{};
		m_deviceInfo->sensorSpec = applied_spec;
		m_deviceInfoUpdated = true;
		m_connectionUpdated.store(false);
	}

	std::lock_guard lock(m_clientMutex);
	if (!m_client)
	{
		m_lastFrame = retina::Frame{};
		return;
	}

	m_client->getFrames([this](const std::deque<retina::Frame>& frames)
		{
			if (frames.empty())
				return;

			m_lastFrame = frames.back();
			m_lastBandwidthMbps = static_cast<float>(m_client->getBandwidthMbps());
			m_lastFrameRate = static_cast<float>(m_client->getFrameRate());

			if (m_lastFrame.deltaUs == 0)
				m_lastFrame.deltaUs = 50000;
		});
}

void MyApp::drawFrame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);

	ImGuiContext& g = *GImGui;
	g.NextWindowData.MenuBarOffsetMinVal.x = g.Style.FramePadding.x * 2.0f;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

	const ImGuiWindowFlags window_flags =
		ImGuiWindowFlags_MenuBar |
		ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoDecoration |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoBringToFrontOnFocus |
		ImGuiWindowFlags_NoNavFocus |
		ImGuiWindowFlags_NoScrollWithMouse;

	ImGui::Begin("RadarDockHost", nullptr, window_flags);
	ImGui::PopStyleVar(3);

	drawMenu();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	const ImGuiID dockspace_id = ImGui::GetID("RadarDockSpace");
	const float footer_height = 42.0f;
	const ImVec2 avail = ImGui::GetContentRegionAvail();
	const ImVec2 dockspace_size(avail.x, std::max(1.0f, avail.y - footer_height));
	ImGui::DockSpace(dockspace_id, dockspace_size);
	ImGui::PopStyleVar();

	if (!m_dockLayoutInitialized)
	{
		if (!m_hasSavedDockLayout || ImGui::DockBuilderGetNode(dockspace_id) == nullptr)
			setupDockLayout(dockspace_id);
		m_dockLayoutInitialized = true;
	}

	drawFooter(footer_height);
	ImGui::End();

	for (auto& wnd : m_allWindows)
	{
		if (!wnd->isOpen())
			continue;

		wnd->draw(ImVec2(-1.0f, -1.0f));
		if (wnd->isActive())
			m_activeWindow = wnd.get();
	}

	ImGui::Render();

	int display_width = 0;
	int display_height = 0;
	glfwGetFramebufferSize(m_window, &display_width, &display_height);
	glViewport(0, 0, display_width, display_height);
	glClearColor(0.1f, 0.1f, 0.1f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_window);
}

void MyApp::drawMenu()
{
	bool open_about = false;

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu(ICON_FA_FILE "  File"))
		{
			if (MyGui::IconMenu(ICON_FA_FILE, "New Recording"));
			if (MyGui::IconMenu(ICON_FA_FOLDER_OPEN, "Open Recording"));
			if (MyGui::IconMenu(ICON_FA_FLOPPY_DISK, "Save Recording"));
			if (MyGui::IconMenu(ICON_FA_FLOPPY_DISK, "Save Recording As..."));

			ImGui::Separator();
			if (MyGui::IconMenu(ICON_FA_POWER_OFF, "Exit"))
				glfwSetWindowShouldClose(m_window, true);

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(ICON_FA_WINDOW_RESTORE "  Window"))
		{
			MyGui::IconMenu(ICON_FA_TERMINAL, "Console", nullptr, &m_consoleWindow->isOpenRef());
			MyGui::IconMenu(ICON_FA_SATELLITE_DISH, "Sensor", nullptr, &m_sensorWindow->isOpenRef());
			MyGui::IconMenu(ICON_FA_FILM, "Recording", nullptr, &m_recordingWindow->isOpenRef());

			ImGui::Separator();
			for (int i = 0; i < static_cast<int>(m_pointWindows.size()); ++i)
			{
				const std::string label = "Window" + std::to_string(i + 1);
				MyGui::IconMenu(ICON_FA_CUBE, label.c_str(), nullptr, &m_pointWindows[i]->isOpenRef());
			}

			ImGui::Separator();
			MyGui::IconMenu(ICON_FA_CHART_COLUMN, "Analysis", nullptr, &m_analysisWindow->isOpenRef());

			ImGui::Separator();
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted("Theme");
			ImGui::SameLine();
			ImGui::SetNextItemWidth(160.0f);
			ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
			if (ImGui::Combo("##Theme", &m_theme, MyGui::themeLabels(), MyGui::ThemeCount))
				setTheme(m_theme);
			ImGui::PopItemFlag();

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(ICON_FA_WIFI "  Sensor"))
		{
			const auto& network_info = getLocalNetworkInfo();
			const auto& device_info = getDeviceInfo();

			ImGui::Text("IP        : %s", network_info.address.empty() ? "N/A" : network_info.address.c_str());
			ImGui::Text("Subnet    : %s", network_info.subnetMask.empty() ? "N/A" : network_info.subnetMask.c_str());
			ImGui::Text("Device IP : %s", device_info.ip.empty() ? "N/A" : device_info.ip.c_str());
			ImGui::Text("Device MAC: %s", device_info.mac.empty() ? "N/A" : device_info.mac.c_str());

			const bool can_connect = getConnectionStatus() == ConnectionStatus::Disconnected;

			ImGui::Separator();
			if (MyGui::IconMenu(ICON_FA_PLUG, "Auto Connect", nullptr, false, can_connect))
				connect();

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(ICON_FA_CIRCLE_QUESTION "  Help"))
		{
			MyGui::IconMenu(ICON_FA_CIRCLE_QUESTION, "No Help yet...");

			ImGui::Separator();
			if (MyGui::IconMenu(ICON_FA_CIRCLE_INFO, "About"))
				open_about = true;

			ImGui::SeparatorText("UI Performance");
			ImGui::Text("Application %.3f ms", 1000.0f / ImGui::GetIO().Framerate);
			ImGui::Text("frame (%.1f FPS)", ImGui::GetIO().Framerate);
			ImGui::Spacing();

			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}

	if (open_about)
		ImGui::OpenPopup("AboutPopup");

	if (ImGui::BeginPopupModal("AboutPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::Text("Radar Studio");
		ImGui::Text("Radar Point Cloud Visualization Tool");
		ImGui::Text("Version 1.0.0");
		ImGui::Separator();
		ImGui::Text("Developed by dandevlog0206");
		ImGui::Text("2026");
		ImGui::Separator();
		ImGui::TextUnformatted("Target pictograms from Flaticon (www.flaticon.com)");
		ImGui::TextUnformatted("created by Stockio, Magnific, meaicon, Leremy");
		if (ImGui::Button("Close"))
			ImGui::CloseCurrentPopup();
		ImGui::EndPopup();
	}
}

void MyApp::drawFooter(float height)
{
	ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetColorU32(ImVec4(0.07f, 0.08f, 0.09f, 1.0f)));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 3.0f));
	if (ImGui::BeginChild("##GlobalFooter", ImVec2(0.0f, height), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
	{
		const float window_width = ImGui::GetWindowContentRegionMax().x;
		const float right_group_width = 160.0f;
		const float right_x = std::max(ImGui::GetCursorStartPos().x, window_width - right_group_width);

		ImGui::SetCursorPos(ImVec2(right_x, 6.0f));
		drawConnectionStatus();
	}
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();
}

void MyApp::drawConnectionStatus()
{
	static const char* kLoadingText[] = { "", ".", "..", "..." };
	static const float kIconSize = 24.0f;
	static const ImU32 s_red = ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
	static const ImU32 s_green = ImGui::GetColorU32(ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
	static const ImU32 s_yellow = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
	static float s_t = 0.0f;

	const auto status = getConnectionStatus();

	const char* status_text = nullptr;
	const char* icon = ICON_FA_PLUG;
	ImVec4 text_color;
	ImU32 icon_color;

	switch (status)
	{
	case ConnectionStatus::Disconnected:
		status_text = "Disconnected";
		text_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
		icon = ICON_FA_LINK_SLASH;
		icon_color = s_red;
		s_t = 0.0f;
		break;
	case ConnectionStatus::Finding:
		status_text = "Finding";
		text_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
		icon = ICON_FA_CIRCLE_NOTCH;
		icon_color = s_yellow;
		break;
	case ConnectionStatus::Connecting:
		status_text = "Connecting";
		text_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
		icon = ICON_FA_SPINNER;
		icon_color = s_yellow;
		break;
	case ConnectionStatus::Connected:
		status_text = "Connected";
		text_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
		icon = ICON_FA_PLUG_CIRCLE_CHECK;
		icon_color = s_green;
		s_t = 0.0f;
		break;
	case ConnectionStatus::Failed:
		status_text = "Failed";
		text_color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
		icon = ICON_FA_CIRCLE_XMARK;
		icon_color = s_red;
		s_t = 0.0f;
		break;
	}

	MyGui::Icon(icon, kIconSize, icon_color);
	ImGui::SameLine();
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4.0f);
	ImGui::TextColored(text_color, "%s%s", status_text, kLoadingText[(int)(s_t * 5) % 4]);

	s_t = fmodf(s_t + ImGui::GetIO().DeltaTime, 10.0f);
}

void MyApp::limitFrameRate(float fps) const
{
	const float total_time = 1.0f / fps;
	const float delay_time = total_time - ImGui::GetIO().DeltaTime;
	if (delay_time <= 0.0f)
		return;

	const auto delay_us = static_cast<uint64_t>(delay_time * 1'000'000.0f);
	std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

void MyApp::connectToDeviceAsync(const std::string& host)
{
	m_connectionStatus.store(ConnectionStatus::Connecting);

	{
		std::lock_guard<std::mutex> lock(m_clientMutex);
		m_client = std::make_unique<retina::DeviceClient>(*m_io, host, log());
		m_client->setOnConnected([this]()
			{
				m_connectionStatus.store(ConnectionStatus::Connected);
				m_connectionUpdated.store(true);
			});
		m_client->setOnFrame([this](const retina::Frame& frame)
			{
				if (m_liveRelay)
					m_liveRelay->send(frame);   // called on the network thread; LiveRelay locks internally
			});
	}

	try
	{
		m_io->run();
	}
	catch (const std::exception& e)
	{
		{
			std::lock_guard<std::mutex> lock(m_clientMutex);
			m_client.reset();
		}

		m_connectionStatus.store(ConnectionStatus::Failed);
		log() << "[fatal] network: " << e.what() << "\n";
	}

	{
		std::lock_guard<std::mutex> lock(m_clientMutex);
		m_client.reset();
	}

	m_connectionStatus.store(ConnectionStatus::Disconnected);
	m_connectionUpdated.store(true);
	m_io->restart();
}

std::ostream& MyApp::log()
{
	return m_console.getOutput();
}

void MyApp::glfwErrorCallback(int error, const char* description)
{
	throw std::runtime_error("[glfw] error: " + std::to_string(error) + ": " + description);
}

void MyApp::glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos)
{
	auto& app = *reinterpret_cast<MyApp*>(glfwGetWindowUserPointer(window));

	if (!app.m_windowMinimized && !app.m_windowMaximized)
	{
		app.m_windowRestored.x = xpos;
		app.m_windowRestored.y = ypos;
	}

	app.m_windowCurrent.x = xpos;
	app.m_windowCurrent.y = ypos;
}

void MyApp::glfwWindowSizeCallback(GLFWwindow* window, int width, int height)
{
	auto& app = *reinterpret_cast<MyApp*>(glfwGetWindowUserPointer(window));

	if (!app.m_windowMinimized && !app.m_windowMaximized)
	{
		app.m_windowRestored.width = width;
		app.m_windowRestored.height = height;
	}

	app.m_windowCurrent.width = width;
	app.m_windowCurrent.height = height;
}

void MyApp::glfwWindowIconifyCallback(GLFWwindow* window, int iconified)
{
	auto& app = *reinterpret_cast<MyApp*>(glfwGetWindowUserPointer(window));
	app.m_windowMinimized = (iconified != 0);
}

void MyApp::glfwWindowMaximizeCallback(GLFWwindow* window, int maximized)
{
	auto& app = *reinterpret_cast<MyApp*>(glfwGetWindowUserPointer(window));
	app.m_windowMaximized = (maximized != 0);
}

void* MyApp::windowReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name)
{
	(void)ctx;
	if (strcmp(name, "GLFW_Window") != 0)
		return nullptr;
	return handler->UserData;
}

void MyApp::windowReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line)
{
	(void)ctx;
	(void)entry;

	int x, y, w, h, value;
	auto& app = *static_cast<MyApp*>(handler->UserData);

	if (sscanf(line, "Pos=%d,%d", &x, &y) == 2)
	{
		app.m_windowCurrent.x = x;
		app.m_windowCurrent.y = y;
		app.m_windowRestored.x = x;
		app.m_windowRestored.y = y;
	}
	else if (sscanf(line, "Size=%d,%d", &w, &h) == 2)
	{
		app.m_windowCurrent.width = w;
		app.m_windowCurrent.height = h;
		app.m_windowRestored.width = w;
		app.m_windowRestored.height = h;
	}
	else if (sscanf(line, "Minimized=%d", &value) == 1)
	{
		app.m_windowMinimized = (value != 0);
	}
	else if (sscanf(line, "Maximized=%d", &value) == 1)
	{
		app.m_windowMaximized = (value != 0);
	}
}

void MyApp::windowWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	(void)ctx;

	auto& app = *static_cast<MyApp*>(handler->UserData);
	buf->appendf("[%s][%s]\n", handler->TypeName, "GLFW_Window");
	buf->appendf("Pos=%d,%d\n", app.m_windowRestored.x, app.m_windowRestored.y);
	buf->appendf("Size=%d,%d\n", app.m_windowRestored.width, app.m_windowRestored.height);
	buf->appendf("Minimized=%d\n", app.m_windowMinimized ? 1 : 0);
	buf->appendf("Maximized=%d\n", app.m_windowMaximized ? 1 : 0);
	buf->append("\n");
}

void* MyApp::sensorSpecReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name)
{
	(void)ctx;
	if (strcmp(name, "Sensor") != 0)
		return nullptr;
	return handler->UserData;
}

void MyApp::sensorSpecReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line)
{
	(void)ctx;
	(void)entry;

	auto& spec = static_cast<MyApp*>(handler->UserData)->m_deviceInfo->sensorSpec;
	float a, b, c;

	if (sscanf(line, "HFov=%f", &a) == 1)
		spec.hfovDeg = a;
	else if (sscanf(line, "VFov=%f", &a) == 1)
		spec.vfovDeg = a;
	else if (sscanf(line, "Size=%f,%f", &a, &b) == 2)
	{
		spec.sensorWidth = a;
		spec.sensorHeight = b;
	}
	else if (sscanf(line, "Position=%f,%f,%f", &a, &b, &c) == 3)
	{
		spec.posX = a;
		spec.posY = b;
		spec.posZ = c;
	}
	else if (sscanf(line, "Rotation=%f,%f", &a, &b) == 2)
	{
		spec.yawDeg = a;
		spec.pitchDeg = b;
	}
	else if (sscanf(line, "RangeX=%f,%f", &a, &b) == 2)
	{
		spec.rangeMinX = a;
		spec.rangeMaxX = b;
	}
	else if (sscanf(line, "RangeY=%f,%f", &a, &b) == 2)
	{
		spec.rangeMinY = a;
		spec.rangeMaxY = b;
	}
	else if (sscanf(line, "RangeZ=%f,%f", &a, &b) == 2)
	{
		spec.rangeMinZ = a;
		spec.rangeMaxZ = b;
	}
	else if (sscanf(line, "Range=%f", &a) == 1)
		spec.range = a;
}

void MyApp::sensorSpecWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	(void)ctx;

	const auto& spec = static_cast<MyApp*>(handler->UserData)->m_deviceInfo->sensorSpec;
	buf->appendf("[%s][%s]\n", handler->TypeName, "Sensor");
	buf->appendf("HFov=%g\n", spec.hfovDeg);
	buf->appendf("VFov=%g\n", spec.vfovDeg);
	buf->appendf("Size=%g,%g\n", spec.sensorWidth, spec.sensorHeight);
	buf->appendf("Position=%g,%g,%g\n", spec.posX, spec.posY, spec.posZ);
	buf->appendf("Rotation=%g,%g\n", spec.yawDeg, spec.pitchDeg);
	buf->appendf("RangeX=%g,%g\n", spec.rangeMinX, spec.rangeMaxX);
	buf->appendf("RangeY=%g,%g\n", spec.rangeMinY, spec.rangeMaxY);
	buf->appendf("RangeZ=%g,%g\n", spec.rangeMinZ, spec.rangeMaxZ);
	buf->appendf("Range=%g\n", spec.range);
	buf->append("\n");
}

void MyApp::setTheme(int theme)
{
	theme = std::clamp(theme, 0, MyGui::ThemeCount - 1);
	m_theme = theme;
	MyGui::applyTheme(static_cast<MyGui::Theme>(m_theme));
	if (ImGui::GetCurrentContext() != nullptr)
		ImGui::MarkIniSettingsDirty();
}

void* MyApp::themeReadOpen(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name)
{
	(void)ctx;
	if (strcmp(name, "Settings") != 0)
		return nullptr;
	return handler->UserData;
}

void MyApp::themeReadLine(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line)
{
	(void)ctx;
	(void)entry;

	char theme_name[64] = {};
	if (sscanf(line, "Theme=%63s", theme_name) == 1)
		static_cast<MyApp*>(handler->UserData)->m_theme = static_cast<int>(MyGui::themeFromKey(theme_name));
}

void MyApp::themeWriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	(void)ctx;

	auto& app = *static_cast<MyApp*>(handler->UserData);
	buf->appendf("[%s][%s]\n", handler->TypeName, "Settings");
	buf->appendf("Theme=%s\n", MyGui::themeKey(static_cast<MyGui::Theme>(app.m_theme)));
	buf->append("\n");
}

int main(int, char**)
{
	try
	{
		MyApp app;
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << "[fatal] exception: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
