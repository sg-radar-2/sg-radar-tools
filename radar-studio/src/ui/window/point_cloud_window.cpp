#include "point_cloud_window.h"
#include "../implot3d_custom.h"
#include "../../my_app.h"

#include <common/ui/imgui_custom.h>

#include <implot3d_internal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace
{
	struct ViewLabel
	{
		const char* name;
		ImPlot3DQuat rot;
	};

	bool check_same_frame_fast(const retina::Frame& a, const retina::Frame& b)
	{
		return a.packetSize == b.packetSize &&
			a.frameCount == b.frameCount &&
			a.deltaUs == b.deltaUs &&
			a.points.size() == b.points.size() &&
			a.targets.size() == b.targets.size();
	}

	ImU64 fnv1a(ImU64 hash, const void* data, size_t size)
	{
		const auto* bytes = static_cast<const unsigned char*>(data);
		for (size_t i = 0; i < size; ++i)
		{
			hash ^= bytes[i];
			hash *= 1099511628211ull;
		}
		return hash;
	}

	template <typename T>
	ImU64 fnv1a_value(ImU64 hash, const T& value)
	{
		return fnv1a(hash, &value, sizeof(value));
	}

	constexpr ImPlot3DQuat kDefaultViewRotation(-0.513269, -0.212596, -0.318184, 0.76819);
	constexpr float kTargetPictogramSize = 48.0f;
	constexpr double kTargetPictogramHeightSmoothing = 0.3; // seconds (time constant)

	// TODO: split fast-doppler Walking into a Running pictogram.
	MyApp::Pictogram get_target_pictogram(const retina::Target& target)
	{
		switch (target.status)
		{
		case retina::TargetStatus::Sitting: return MyApp::Pictogram::Sitting;
		case retina::TargetStatus::Lying: return MyApp::Pictogram::Lying;
		case retina::TargetStatus::Walking: return MyApp::Pictogram::Walking;
		case retina::TargetStatus::Standing:
		default: return MyApp::Pictogram::Standing;
		}
	}

	ImPlot3DBox make_axes_box(const retina::SensorSpec& spec)
	{
		return ImPlot3DBox(
			ImPlot3DPoint(spec.rangeMinX, spec.rangeMinY, spec.rangeMinZ - 0.1f),
			ImPlot3DPoint(spec.rangeMaxX, spec.rangeMaxY, spec.rangeMaxZ + 0.1f));
	}

	ImPlot3DQuat get_axis_aligned_view_rotation(ImAxis3D axis, bool positive_direction)
	{
		constexpr double pi = 3.14159265358979323846;
		constexpr double half_pi = pi * 0.5;
		constexpr ImPlot3DPoint x_axis(1.f, 0.f, 0.f);
		constexpr ImPlot3DPoint y_axis(0.f, 1.f, 0.f);
		constexpr ImPlot3DPoint z_axis(0.f, 0.f, 1.f);
	
		if (axis == ImAxis3D_X && positive_direction)
			return ImPlot3DQuat(half_pi, y_axis) * ImPlot3DQuat(-half_pi, x_axis); // Left(+x)
		else if (axis == ImAxis3D_X && !positive_direction)
			return ImPlot3DQuat(-half_pi, z_axis) * ImPlot3DQuat(-half_pi, y_axis); // Right(-x)
		else if (axis == ImAxis3D_Y && positive_direction)
			return ImPlot3DQuat(-half_pi, x_axis); // Rear(+y)
		else if (axis == ImAxis3D_Y && !positive_direction)
			return ImPlot3DQuat(pi, z_axis) * ImPlot3DQuat(half_pi, x_axis); // Front(-y)
		else if (axis == ImAxis3D_Z && positive_direction)
			return ImPlot3DQuat(pi, z_axis) * ImPlot3DQuat(pi, x_axis); // Bottom(+z)
		else if (axis == ImAxis3D_Z && !positive_direction)
			return ImPlot3DQuat(); // Top(-z)
	
		return ImPlot3DQuat();
	}
}

PointCloudWindow::PointCloudWindow(const std::string& title) :
	DockingWindow(title, true),
	m_plotTitle(title + "##Plot"),
	m_plotLabel("##" + title + "_Plot"),
	m_gizmoLabel("##" + title + "_Gizmo"),
	m_sliderLabel("##" + title + "_Slider"),
	m_viewRotation(kDefaultViewRotation),
	m_viewAxesBox(make_axes_box(retina::SensorSpec{})),
	m_viewZoom(1.0),
	m_source(WindowSource::None),
	m_sessionId(0),
	m_playbackIdx(0),
	m_speedIdx(3),
	m_playbackSpeed(1.0),
	m_playbackTimer(0.0),
	m_playbackAccumulated(0.0),
	m_isPlaying(false),
	m_isRecording(false)
{
	setWindowFlags(
		ImGuiWindowFlags_MenuBar |
		ImGuiWindowFlags_NoScrollbar |
		ImGuiWindowFlags_NoScrollWithMouse);
}

void PointCloudWindow::setRecordingWindow(RecordingWindow* window)
{
	m_recordingWindow = window;
}

void PointCloudWindow::setSensorSpec(const retina::SensorSpec& spec)
{
	m_sensorSpec = spec;
	m_viewAxesBox = make_axes_box(spec);
	m_viewZoom = 1.0;
}

const retina::Frame& PointCloudWindow::getLastFrame() const
{
	return MyApp::get().getLastFrame();
}

void PointCloudWindow::onUpdate()
{
	if (m_source == WindowSource::None)
	{
		m_sessionId = -1;
		m_currSession = nullptr;
		m_currFrame = nullptr;
		m_playbackIdx = 0;

		return;
	}
	if (m_source == WindowSource::Sensor)
	{
		m_currFrame = &MyApp::get().getLastFrame();
	}
	else if (m_source == WindowSource::CurrentSession || m_source == WindowSource::Session)
	{
		if (m_source == WindowSource::CurrentSession)
		{
			auto* curr_session = m_recordingWindow->getCurrentSession();
			if (curr_session != m_currSession)
			{
				m_sessionId = curr_session ? curr_session->id : -1;
				m_currSession = curr_session;
				m_currFrame = nullptr;
				m_isPlaying = false;
				m_playbackIdx = 0;

				setSensorSpec(m_recordingWindow->getRecorder().getSensorSpec());
			}

		}
		else if (m_source == WindowSource::Session)
			m_currSession = m_recordingWindow->getRecorder().getSession(m_sessionId);

		if (m_currSession == nullptr)
		{
			m_currFrame = nullptr;
			m_sessionId = -1;
			resetPlaybackState();
		}
		else if (m_isRecording)
		{
			m_currFrame = &MyApp::get().getLastFrame();
			resetPlaybackState();
		}
		else if (m_isPlaying)
		{
			updatePlaybackState();
			if (m_currSession != nullptr && !m_currSession->frames.empty())
				m_currFrame = &m_currSession->frames[m_playbackIdx].frame;
		}
		else
		{
			const auto frame_count = static_cast<int>(m_currSession->frames.size());

			if (frame_count > 0)
			{
				m_playbackIdx = std::clamp(m_playbackIdx, 0, frame_count - 1);
				m_currFrame = &m_currSession->frames[m_playbackIdx].frame;
			}
			else
			{
				m_playbackIdx = 0;
				m_currFrame = nullptr;
			}
			resetPlaybackState();
		}
	}
}

void PointCloudWindow::onDraw(const ImVec2& size)
{
	onDrawMenuBar();

	if (m_source == WindowSource::None)
	{
		auto& plot_style = ImPlot3D::GetStyle();
		auto* draw_list = ImGui::GetWindowDrawList();

		const ImVec2 padding = plot_style.PlotPadding;
		const ImVec2 window_pos = ImGui::GetWindowPos();
		const ImVec2 window_size = ImGui::GetWindowSize();

		// compute absolute rectangle (respecting padding)
		const ImVec2 p_min = ImVec2(window_pos.x + padding.x, window_pos.y + padding.y);
		const ImVec2 p_max = ImVec2(window_pos.x + window_size.x - padding.x, window_pos.y + window_size.y - padding.y);
		const ImU32 bg_color = ImGui::GetColorU32(plot_style.Colors[ImPlot3DCol_FrameBg]);

		draw_list->AddRectFilled(p_min, p_max, bg_color);
		drawCenteredText("No Source Selected");

		return;
	}

	const bool show_playback_controls = m_source == WindowSource::CurrentSession || m_source == WindowSource::Session;
	const ImVec2 plot_avail = ImGui::GetContentRegionAvail();
	const float plot_width = std::max(1.0f, plot_avail.x);
	const float controls_height = show_playback_controls ? 70.0f : 0.0f;
	const float plot_height = std::max(180.0f, plot_avail.y - controls_height);
	const ImVec2 plot_size(plot_width, plot_height);
	const ImVec2 plot_pos = ImGui::GetCursorScreenPos();

	const bool keyboard_focus = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
	const bool keyboard_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
	const bool keyboard_active = keyboard_focus || keyboard_hovered;

	if (keyboard_active)
	{
		if (ImGui::IsKeyPressed(ImGuiKey_Space, false))
			onPlayButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))
			onRecordButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false))
			onTrimLeftButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, false))
			onTrimRightButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_Minus, false))
			m_recordingWindow->onDeleteSessionButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_Equal, false))
			m_recordingWindow->onAddSessionButtonClicked();
		if (ImGui::IsKeyPressed(ImGuiKey_Backspace, false))
		{
			auto& recorder = m_recordingWindow->getRecorder();
			recorder.clearSession(m_sessionId);
		}
		if (ImGui::IsKeyPressed(ImGuiKey_Backslash, false))
		{
			auto& recorder = m_recordingWindow->getRecorder();
			if (!m_isRecording && (recorder.empty() || !recorder.getSession(m_sessionId)->frames.empty()))
				m_recordingWindow->onAddSessionButtonClicked();
			onRecordButtonClicked();
		}
	}

	if (m_isRecording)
	{
		auto& recorder = m_recordingWindow->getRecorder();
		size_t frame_count = recorder.getSession(m_sessionId)->frames.size();
		if (frame_count >= static_cast<size_t>(m_recordMaxFrameLength))
		{
			onRecordButtonClicked();
		}
	}

	static const auto kPlotFlags = ImPlot3DFlags_NoZoom;
	if (ImPlot3D::BeginPlot(m_plotTitle.c_str(), plot_size, kPlotFlags))
	{
		ImPlot3D::SetupAxis(ImAxis3D_X, "X");
		ImPlot3D::SetupAxis(ImAxis3D_Y, "Y");
		ImPlot3D::SetupAxis(ImAxis3D_Z, "Z");

		if (m_showGizmo)
		{
			MyPlot3D::SetViewGizmoInitialRotation(m_gizmoLabel.c_str(), kDefaultViewRotation);
			MyPlot3D::SetViewGizmoInitialAxesBox(m_gizmoLabel.c_str(), make_axes_box(m_sensorSpec));
			MyPlot3D::SetViewGizmoInitialZoom(m_gizmoLabel.c_str(), 1.0);
			MyPlot3D::ShowViewGizmo(m_gizmoLabel.c_str(), 85.0f, &m_viewRotation, &m_viewAxesBox, &m_viewZoom);
		}

		if (m_currFrame != nullptr)
		{
			static const auto getter = [](void* user, const void* point, ImPlot3DPoint* out_point, ImU32* out_color)
				{
					const auto& wnd = *reinterpret_cast<PointCloudWindow*>(user);
					const auto& p = *reinterpret_cast<const retina::Point*>(point);
					
					out_point->x = p.x;
					out_point->y = p.y;
					out_point->z = p.z;
					
					*out_color = wnd.getPointColor(p.doppler, p.power);
				};

			ImPlot3DSpec spec;
			spec.Marker = ImPlot3DMarker_Circle;
			spec.MarkerSize = m_visualSetting.pointSize;
			spec.FillAlpha = 1.0f;
			spec.Flags = ImPlot3DItemFlags_NoLegend;

			MyPlot3D::PlotScatter(
				m_plotLabel.c_str(),
				this,
				m_currFrame->points.data(),
				static_cast<int>(m_currFrame->points.size()),
				static_cast<int>(sizeof(retina::Point)),
				getter,
				spec,
				makeScatterContentKey());

			if (m_showTargets)
			{
				spec.Marker = ImPlot3DMarker_None;
				spec.FillAlpha = 0.1f;
				for (const auto& target : m_currFrame->targets)
				{
					if (m_showTargetPictogram && queueTargetPictogram(target))
						continue;
					drawTargetBox(target, spec);
				}
			}
		}

		if (m_frustumType != 0)
			drawFrustum(m_frustumType == 2);

		ImPlot3D::EndPlot();
		drawQueuedPictograms();
	}

	if (m_source == WindowSource::Sensor)
	{
		if (MyApp::get().getConnectionStatus() == MyApp::ConnectionStatus::Disconnected)
			drawCenteredText("Sensor Disconnected");
	}
	else if (m_source == WindowSource::CurrentSession)
	{
		if (m_currSession == nullptr)
			drawCenteredText("No Current Session");
		else if (m_currSession->frames.empty())
			drawCenteredText("Empty Session");
	}
	else if (m_source == WindowSource::Session)
	{
		if (m_currSession == nullptr)
			drawCenteredText("No Session Selected");
		else if (m_currSession->frames.empty())
			drawCenteredText("Empty Session");
	}

	if (show_playback_controls)
	{
		if (keyboard_active && m_currSession != nullptr)
		{
			if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false))
			{
				m_playbackIdx = std::max(0, m_playbackIdx - 1);
				resetPlaybackState();
			}
			if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false))
			{
				const int frame_count = static_cast<int>(m_currSession->frames.size());
				m_playbackIdx = std::min(frame_count - 1, m_playbackIdx + 1);
				resetPlaybackState();
			}
			if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false))
			{
				m_playbackIdx = std::max(0, m_playbackIdx + 20);
				resetPlaybackState();
			}
			if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false))
			{
				const int frame_count = static_cast<int>(m_currSession->frames.size());
				m_playbackIdx = std::min(frame_count - 1, m_playbackIdx + 20);
				resetPlaybackState();
			}
		}

		drawPlaybackControls();

		// Position record indicator in the top-left corner of the plot area
		const ImVec2 record_indicator_pos = ImVec2(plot_pos.x + 12.0f, plot_pos.y + 12.0f);
		ImGui::SetCursorScreenPos(record_indicator_pos);
		MyGui::RecordIndicator("##RecordIndicator", m_isRecording, ImVec2(44.0f, 22.0f));

		if (m_isRecording && m_currSession)
		{
			const float thickness = 2.0f;
			const float padding = 3.0f;
			const ImVec2 rect_min(plot_pos.x + padding, plot_pos.y + padding);
			const ImVec2 rect_max(plot_pos.x + plot_size.x - padding, plot_pos.y + plot_size.y - padding);

			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRect(rect_min, rect_max, IM_COL32(255, 0, 0, 255), 1.0f, 0, thickness);

			// TODO: improve logic
			// Actually record the frame here
			if (m_currSession->frames.empty())
			{
				auto frame = PointCloudRecorder::RecordFrame{ MyApp::get().getLastFrame() };
				auto& recorder = m_recordingWindow->getRecorder();
				recorder.appendFrame(m_currSession->id, frame);
				m_playbackIdx = 0;
			}
			else
			{
				const auto& curr_frame = m_currSession->frames[m_playbackIdx].frame;
				const auto& new_frame = MyApp::get().getLastFrame();

				// Check if the frame is really new
				if (!check_same_frame_fast(curr_frame, new_frame))
				{
					auto frame = PointCloudRecorder::RecordFrame{ new_frame };
					auto& recorder = m_recordingWindow->getRecorder();
					
					recorder.insertFrame(m_currSession->id, frame, m_playbackIdx + 1);

					if (m_playbackIdx < static_cast<int>(m_currSession->frames.size()) - 1)
						m_playbackIdx++;
				}
			}
		}
	}

	drawOrientationLabel();
}

void PointCloudWindow::onDrawMenuBar()
{
	if (!ImGui::BeginMenuBar())
		return;

	if (ImGui::BeginMenu("Source"))
	{
		static const char* kSourceItems = "None\0Sensor\0Current Session\0Session\0";
		int source_idx = static_cast<int>(m_source);

		ImGui::SetNextItemWidth(210.0f);
		if (ImGui::Combo("##FooterSource", &source_idx, kSourceItems))
		{
			m_source = static_cast<WindowSource>(source_idx);
			if (m_source == WindowSource::Session)
				m_playbackIdx = 0;
		}

		ImGui::Separator();
		if (m_source == WindowSource::Session)
		{
			const auto& recorder = m_recordingWindow->getRecorder();

			if (m_sessionId != -1)
			{
				const auto& selected_item = recorder.getSession(m_sessionId);

				ImGui::SameLine();
				ImGui::SetNextItemWidth(400.0f);
				if (ImGui::BeginCombo("##FooterRecording", selected_item->name.c_str()))
				{
					const int session_count = recorder.getSessionCount();

					for (int i = 0; i < session_count; ++i)
					{
						const auto id = recorder.getOrderedSessionID(i);
						const auto& item = recorder.getSession(id);

						if (ImGui::Selectable(item->name.c_str(), id == m_sessionId))
						{
							m_sessionId = id;
							m_playbackIdx = 0;
						}
					}

					ImGui::EndCombo();
				}
			}
		}

		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("View"))
	{
		if (MyGui::IconMenu(ICON_FA_ROTATE, "Reset View"))
			m_viewRotation = kDefaultViewRotation;

		ImGui::Separator();
		if (MyGui::IconMenu(ICON_FA_ARROW_LEFT, "Left"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_X, true);
		if (MyGui::IconMenu(ICON_FA_ARROW_RIGHT, "Right"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_X, false);
		if (MyGui::IconMenu(ICON_FA_ARROW_DOWN, "Rear"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_Y, true);
		if (MyGui::IconMenu(ICON_FA_ARROW_UP, "Front"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_Y, false);
		if (MyGui::IconMenu(ICON_FA_ARROW_DOWN, "Bottom"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_Z, true);
		if (MyGui::IconMenu(ICON_FA_ARROW_UP, "Top"))
			m_viewRotation = get_axis_aligned_view_rotation(ImAxis3D_Z, false);

		ImGui::Separator();
		ImGui::SetNextItemWidth(130.0f);
		ImGui::Combo("Frustum", &m_frustumType, "None\0Wireframe\0Polygon\0");

		ImGui::Separator();
		MyGui::IconMenu(ICON_FA_BULLSEYE, "Targets", nullptr, &m_showTargets);
		MyGui::IconMenu(ICON_FA_PERSON, "Target Pictogram", nullptr, &m_showTargetPictogram, m_showTargets);
		MyGui::IconMenu(ICON_FA_COMPASS, "Gizmo", nullptr, &m_showGizmo);

		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Visualize"))
	{
		ImGui::TextUnformatted("Doppler -> Color");
		MyGui::LegendBar(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
		ImGui::SetNextItemWidth(200.f);
		ImGui::DragFloatRange2("##DopplerRange", &m_visualSetting.dopplerMin, &m_visualSetting.dopplerMax, 0.01f, -10.0f, 10.0f, "%.2f", "%.2f");

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::TextUnformatted("Power -> Alpha");
		MyGui::LegendBar(ImVec4(0.f, 0.f, 0.0f, 1.f), ImVec4(1.f, 1.f, 1.0f, 1.0f));
		ImGui::SetNextItemWidth(200.f);
		ImGui::DragFloatRange2("##PowerRange", &m_visualSetting.powerDBMin, &m_visualSetting.powerDBMax, 1.0f, 20.0f, 70.0f, "%.0f", "%.0f");
		ImGui::TextUnformatted("Cutoff Power");
		ImGui::Checkbox("##PowerCutoff", &m_visualSetting.powerCutoffEnabled);

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::TextUnformatted("Point Size");
		ImGui::SetNextItemWidth(200.f);
		ImGui::DragFloat("##PointSize", &m_visualSetting.pointSize, 0.1f, 0.1f, 20.0f, "%.1f");
		ImGui::Separator();
		ImGui::Checkbox("All White", &m_visualSetting.allWhite);

		ImGui::Separator();
		if (ImGui::Button("reset"))
			m_visualSetting = VisualizeSetting{};

		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Record"))
	{
		ImGui::TextUnformatted("Max Frame Length");
		ImGui::InputInt("##MaxFrameLength", &m_recordMaxFrameLength, 1, UINT32_MAX);
		ImGui::EndMenu();
	}

	ImGui::EndMenuBar();
}

void PointCloudWindow::onPlayButtonClicked()
{
	if (m_source != WindowSource::CurrentSession && m_source != WindowSource::Session)
		return;
	if (m_currSession == nullptr || m_currSession->frames.size() < 2 || m_isRecording)
		return;

	if (!m_isPlaying)
	{
		resetPlaybackState();
		m_isPlaying = true;

		if (m_playbackIdx == static_cast<int>(m_currSession->frames.size()) - 1)
			m_playbackIdx = 0;
	}
	else
	{
		m_isPlaying = false;
	}
}

void PointCloudWindow::onRecordButtonClicked()
{
	if (m_source != WindowSource::CurrentSession && m_source != WindowSource::Session)
		return;
	if (m_currSession == nullptr || m_isPlaying)
		return;

	m_isRecording = !m_isRecording;
}

void PointCloudWindow::onTrimLeftButtonClicked()
{
	if (m_source != WindowSource::CurrentSession && m_source != WindowSource::Session)
		return;
	if (m_currSession == nullptr || m_currSession->frames.empty() || m_isPlaying || m_isRecording)
		return;

	if (m_playbackIdx > 0)
	{
		auto& recorder = m_recordingWindow->getRecorder();
		recorder.removeFrame(m_sessionId, 0, m_playbackIdx);
		m_playbackIdx = 0;
	}
}

void PointCloudWindow::onTrimRightButtonClicked()
{
	if (m_source != WindowSource::CurrentSession && m_source != WindowSource::Session)
		return;
	if (m_currSession == nullptr || m_currSession->frames.empty() || m_isPlaying || m_isRecording)
		return;

	int frame_count = static_cast<int>(m_currSession->frames.size());
	if (m_playbackIdx < frame_count - 1)
	{
		auto& recorder = m_recordingWindow->getRecorder();
		recorder.removeFrame(m_sessionId, m_playbackIdx + 1, frame_count - m_playbackIdx - 1);
		m_playbackIdx = frame_count - 1;
	}
}

void PointCloudWindow::onSettingReadLine(void* entry, const char* line)
{
	int value;

	if (strcmp(line, "Source=None") == 0)
		m_source = WindowSource::None;
	else if (strcmp(line, "Source=Sensor") == 0)
		m_source = WindowSource::Sensor;
	else if (strcmp(line, "Source=CurrentSession") == 0)
		m_source = WindowSource::CurrentSession;
	else if (strcmp(line, "Source=Session") == 0)
		m_source = WindowSource::Session;
	else if (sscanf(line, "ShowFrustum=%d", &value) == 1)
		m_frustumType = value;
	else if (sscanf(line, "ShowTargets=%d", &value) == 1)
		m_showTargets = value != 0;
	else if (sscanf(line, "ShowTargetPictogram=%d", &value) == 1)
		m_showTargetPictogram = value != 0;
	else if (sscanf(line, "ShowGizmo=%d", &value) == 1)
		m_showGizmo = value != 0;
}

void PointCloudWindow::onSettingWriteAll(ImGuiTextBuffer* buf) const
{
	switch (m_source)
	{
	case WindowSource::None: buf->append("Source=None\n"); break;
	case WindowSource::Sensor: buf->append("Source=Sensor\n"); break;
	case WindowSource::CurrentSession: buf->append("Source=CurrentSession\n"); break;
	case WindowSource::Session: buf->append("Source=Session\n"); break;
	}
	buf->appendf("ShowTargets=%d\n", static_cast<int>(m_showTargets));
	buf->appendf("ShowTargetPictogram=%d\n", static_cast<int>(m_showTargetPictogram));
	buf->appendf("ShowGizmo=%d\n", static_cast<int>(m_showGizmo));
	buf->appendf("ShowFrustum=%d\n", static_cast<int>(m_frustumType));
}

void PointCloudWindow::drawOrientationLabel()
{
	static const std::array<ViewLabel, 6> kLabels = {
		ViewLabel{ "Left", get_axis_aligned_view_rotation(ImAxis3D_X, true) },
		ViewLabel{ "Right", get_axis_aligned_view_rotation(ImAxis3D_X, false) },
		ViewLabel{ "Rear", get_axis_aligned_view_rotation(ImAxis3D_Y, true) },
		ViewLabel{ "Front", get_axis_aligned_view_rotation(ImAxis3D_Y, false) },
		ViewLabel{ "Bottom", get_axis_aligned_view_rotation(ImAxis3D_Z, true) },
		ViewLabel{ "Top", get_axis_aligned_view_rotation(ImAxis3D_Z, false) },
	};

	for (const auto& item : kLabels)
	{
		const float diff =
			std::abs(m_viewRotation.x - item.rot.x) +
			std::abs(m_viewRotation.y - item.rot.y) +
			std::abs(m_viewRotation.z - item.rot.z) +
			std::abs(m_viewRotation.w - item.rot.w);

		if (diff < 0.02f)
		{
			const auto off_y = m_source == WindowSource::CurrentSession || m_source == WindowSource::Session ? 70.0f : 0.0f;
			const auto text_size = ImGui::CalcTextSize(item.name);
			const auto window_max = ImGui::GetWindowContentRegionMax();
			const auto pos = ImVec2(20.f, window_max.y - text_size.y - off_y - 10.0f);

			ImGui::SetCursorPos(pos);
			ImGui::TextUnformatted(item.name);

			break;
		}
	}
}

void PointCloudWindow::drawPlaybackControls()
{
	const auto& style = ImGui::GetStyle();
	const float spacing = style.ItemInnerSpacing.x;
	const float available_width = std::max(0.0f, ImGui::GetContentRegionAvail().x);
	const float speed_combo_width = ImGui::CalcTextSize("  .00x  ").x;
	const float play_btn_size = 28.0f;
	const float frame_text_width = ImGui::CalcTextSize(" 0000 / 0000 ").x;
	const float slider_width = std::max(120.0f, available_width - speed_combo_width - play_btn_size - frame_text_width - spacing * 4.0f);
	const int frame_count = m_currSession ? static_cast<int>(m_currSession->frames.size()) : 0;
	const bool is_connected = MyApp::get().getConnectionStatus() == MyApp::ConnectionStatus::Connected;
	const bool disable_all = m_currSession == nullptr || m_isPlaying;

	if (frame_count > 0)
		m_playbackIdx = std::clamp(m_playbackIdx, 0, frame_count - 1);
	else
		m_playbackIdx = 0;

	const char* mul_items = ".25x\0"".5x\0""0.75x\0""1x\0""1.5x\0""2x\0""3x\0""4x\0";

	ImGui::SetNextItemWidth(speed_combo_width);
	if (ImGui::Combo("##PlaybackSpeed", &m_speedIdx, mul_items))
	{
		switch (m_speedIdx)
		{
		case 0: m_playbackSpeed = .25; break;
		case 1: m_playbackSpeed = .5; break;
		case 2: m_playbackSpeed = .75; break;
		case 3: m_playbackSpeed = 1.0; break;
		case 4: m_playbackSpeed = 1.5; break;
		case 5: m_playbackSpeed = 2.0; break;
		case 6: m_playbackSpeed = 3.0; break;
		case 7: m_playbackSpeed = 4.0; break;
		}
	}

	ImGui::SameLine();
	ImGui::BeginDisabled(m_isRecording || frame_count == 0);
	auto play_pause_icon = m_isPlaying ? ICON_FA_PAUSE : ICON_FA_PLAY;
	if (MyGui::IconButton("##PlayPause", play_pause_icon, ImVec2(play_btn_size, play_btn_size)))
	{
		assert(m_currSession != nullptr);
		onPlayButtonClicked();
	}
	ImGui::EndDisabled();

	ImGui::BeginDisabled(frame_count <= 1 || m_isRecording);
	ImGui::SameLine();
	ImGui::Dummy(ImVec2(spacing, 0.0f));
	ImGui::SameLine();
	ImGui::SetNextItemWidth(slider_width);
	MyGui::TimelineFrameSlider(m_sliderLabel.c_str(), &m_playbackIdx, 0, std::max(0, frame_count - 1));
	ImGui::EndDisabled();

	ImGui::SameLine();
	char frame_text[64];
	snprintf(frame_text, sizeof(frame_text), " %d / %d ", m_playbackIdx, frame_count > 0 ? frame_count - 1 : 0);

	const float remaining_width = ImGui::GetContentRegionAvail().x;
	const float text_width = ImGui::CalcTextSize(frame_text).x;
	if (remaining_width > text_width + spacing)
	{
		const float offset = (remaining_width - text_width) * 0.5f;
		ImGui::Dummy(ImVec2(offset, 0.0f));
		ImGui::SameLine();
	}
	ImGui::TextUnformatted(frame_text);

	const float action_btn_height = 26.0f;
	const float action_btn_width = 36.0f;
	const float record_btn_width = 60.0f;
	const float pause_btn_width = 54.0f;
	const float stop_btn_width = 38.0f;
	const float single_frame_btn_width = 38.0f;

	const ImVec2 record_btn_size(record_btn_width, action_btn_height);
	const ImVec2 pause_btn_size(pause_btn_width, action_btn_height);
	const ImVec2 stop_btn_size(stop_btn_width, action_btn_height);
	const ImVec2 single_frame_btn_size(single_frame_btn_width, action_btn_height);
	const ImVec2 trim_btn_size(action_btn_width, action_btn_height);
	const ImVec2 delete_btn_size(action_btn_width, action_btn_height);
	const ImVec2 split_btn_size(action_btn_width, action_btn_height);

	auto* recorder = m_recordingWindow ? &m_recordingWindow->getRecorder() : nullptr;

	// Editing
	ImGui::BeginDisabled(disable_all || m_isRecording || !frame_count || m_playbackIdx == 0);
	ImGui::SetNextItemWidth(speed_combo_width);
	if (MyGui::IconButton("##TrimLeft", ICON_FA_CROP, trim_btn_size))
		onTrimLeftButtonClicked();
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || m_isRecording || !frame_count);
	if (MyGui::IconButton("##DeleteFrame", ICON_FA_TRASH_CAN, delete_btn_size))
	{
		recorder->removeFrame(m_sessionId, m_playbackIdx);
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || m_isRecording || !frame_count || m_playbackIdx == frame_count - 1);
	if (MyGui::IconButton("##Split", ICON_FA_CODE_BRANCH, split_btn_size))
	{
		assert(recorder != nullptr);
		recorder->splitSession(m_sessionId, m_playbackIdx);
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || m_isRecording || !frame_count || m_playbackIdx == frame_count - 1);
	if (MyGui::IconButton("##TrimRight", ICON_FA_SCISSORS, trim_btn_size))
		onTrimRightButtonClicked();
	ImGui::EndDisabled();

	// Recording
	const float center_controls_width = record_btn_width + pause_btn_width + stop_btn_width + single_frame_btn_width + 3.0f * spacing;
	const float center_pos = (available_width - center_controls_width) * 0.5f;

	ImGui::SameLine(center_pos > ImGui::GetCursorPosX() ? center_pos : 0.0f);
	ImGui::BeginDisabled(disable_all || !is_connected || m_isRecording);
	if (MyGui::IconButton("##Record", ICON_FA_CIRCLE, record_btn_size, IM_COL32(230, 40, 40, 255)))
	{
		assert(recorder != nullptr);
		recorder->removeFrame(m_sessionId, m_playbackIdx + 1, frame_count - (m_playbackIdx + 1));
		m_isRecording = true;
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || !is_connected);
	if (m_isRecording)
	{
		if (MyGui::IconButton("##Pause", ICON_FA_PAUSE, pause_btn_size))
			m_isRecording = false;
	}
	else
	{
		if (MyGui::IconButton("##Resume", ICON_FA_PLAY, pause_btn_size))
			m_isRecording = true;
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || !is_connected || m_isRecording || (m_isRecording && !frame_count));
	if (MyGui::IconButton("##Stop", ICON_FA_STOP, stop_btn_size))
	{
		assert(recorder != nullptr);
		recorder->removeFrame(m_sessionId, 0, frame_count);
		m_isRecording = false;
		m_playbackIdx = -1;
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(disable_all || !is_connected || m_isRecording);
	if (MyGui::IconButton("##SingleFrame", ICON_FA_FORWARD_STEP, single_frame_btn_size))
	{
		assert(recorder != nullptr);
		PointCloudRecorder::RecordFrame frame = { MyApp::get().getLastFrame() };
		recorder->insertFrame(m_sessionId, frame, m_playbackIdx++);
	}
	ImGui::EndDisabled();

	// Time
	ImGui::SameLine(available_width - 200.0f);
	if (m_currSession)
	{
		uint64_t curr_us = 0;

		// TODO: optimize
		for (int i = 0; i < m_playbackIdx; ++i)
			curr_us += m_currSession->frames[i].frame.deltaUs;

		uint64_t total_ms = (m_currSession->lengthUs / 1000) % 1000;
		uint64_t total_s = (m_currSession->lengthUs / 1000000) % 60;
		uint64_t total_m = (m_currSession->lengthUs / 60000000) % 60;
		uint64_t curr_ms = (curr_us / 1000) % 1000;
		uint64_t curr_s = (curr_us / 1000000) % 60;
		uint64_t curr_m = (curr_us / 60000000) % 60;

		ImGui::Text("%02llu:%02llu.%03llu / %02llu:%02llu.%03llu", curr_m, curr_s, curr_ms, total_m, total_s, total_ms);
	}
	else
		ImGui::TextUnformatted("--:--.---");
}

void PointCloudWindow::drawTargetBox(const retina::Target& target, const ImPlot3DSpec& spec)
{
	const auto text_x = (target.minx + target.maxx) * 0.5f;
	const auto text_y = (target.miny + target.maxy) * 0.5f;
	const auto text_z = target.maxz + 0.3f;
	const std::string text = "ID: " + std::to_string(target.targetId) + '(' + retina::to_string(target.status) + ')';
	ImPlot3D::PlotText(text.c_str(), text_x, text_y, text_z);
	MyPlot3D::PlotBox("##TargetBox", target.minx, target.maxx, target.miny, target.maxy, target.minz, target.maxz, spec);
}

bool PointCloudWindow::queueTargetPictogram(const retina::Target& target)
{
	const unsigned int texture = MyApp::get().getPictogramTexture(get_target_pictogram(target));
	if (texture == 0)
		return false;

	// Use the radar's tracked position for x/y: the box is rebuilt from noisy points every frame and jitters.
	// The packet carries no height, so z comes from the box and is smoothed per target for display.
	const int frame = ImGui::GetFrameCount();
	const bool has_box = target.minz <= target.maxz;
	const double raw_z = has_box ? (target.minz + target.maxz) * 0.5 : 0.0;

	auto it = m_pictogramHeights.find(target.targetId);
	if (it == m_pictogramHeights.end() || it->second.lastFrame < frame - 1)
	{
		it = m_pictogramHeights.insert_or_assign(target.targetId, SmoothedHeight{ raw_z, frame }).first;
	}
	else if (it->second.lastFrame != frame)
	{
		if (has_box)
		{
			const double alpha = 1.0 - std::exp(-ImGui::GetIO().DeltaTime / kTargetPictogramHeightSmoothing);
			it->second.z += (raw_z - it->second.z) * alpha;
		}
		it->second.lastFrame = frame;
	}

	const ImPlot3DPoint anchor(target.x, target.y, it->second.z);

	const ImPlot3DPlot& plot = *ImPlot3D::GetCurrentPlot();
	if (!ImPlot3DBox(plot.RangeMin(), plot.RangeMax()).Contains(anchor))
		return true;

	if (m_queuedPictograms.empty())
	{
		const ImVec2 plot_pos = ImPlot3D::GetPlotRectPos();
		const ImVec2 plot_size = ImPlot3D::GetPlotRectSize();
		m_pictogramClipMin = plot_pos;
		m_pictogramClipMax = ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y);
	}

	m_queuedPictograms.push_back({ ImPlot3D::PlotToPixels(anchor), texture, target.targetId });
	return true;
}

void PointCloudWindow::drawQueuedPictograms()
{
	const int frame = ImGui::GetFrameCount();
	for (auto it = m_pictogramHeights.begin(); it != m_pictogramHeights.end();)
		it = it->second.lastFrame < frame - 1 ? m_pictogramHeights.erase(it) : std::next(it);

	if (m_queuedPictograms.empty())
		return;

	const ImU32 color = ImPlot3D::GetStyleColorU32(ImPlot3DCol_InlayText);
	const ImVec2 half(kTargetPictogramSize * 0.5f, kTargetPictogramSize * 0.5f);

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->PushClipRect(m_pictogramClipMin, m_pictogramClipMax, true);
	for (const QueuedPictogram& pictogram : m_queuedPictograms)
	{
		const ImVec2 icon_min(pictogram.center.x - half.x, pictogram.center.y - half.y);
		const ImVec2 icon_max(pictogram.center.x + half.x, pictogram.center.y + half.y);
		draw_list->AddImage(static_cast<ImTextureID>(pictogram.texture), icon_min, icon_max, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), color);

		const std::string id_text = std::to_string(pictogram.targetId);
		const ImVec2 text_size = ImGui::CalcTextSize(id_text.c_str());
		draw_list->AddText(ImVec2(pictogram.center.x - text_size.x * 0.5f, icon_min.y - text_size.y), color, id_text.c_str());
	}
	draw_list->PopClipRect();

	m_queuedPictograms.clear();
}

void PointCloudWindow::drawFrustum(bool polygon)
{
	constexpr double pi = 3.14159265358979323846;

	const double hfov_deg = static_cast<double>(m_sensorSpec.hfovDeg);
	const double vfov_deg = static_cast<double>(m_sensorSpec.vfovDeg);
	const double sensor_width = static_cast<double>(m_sensorSpec.sensorWidth);
	const double sensor_height = static_cast<double>(m_sensorSpec.sensorHeight);
	const double sensor_pos_x = static_cast<double>(m_sensorSpec.posX);
	const double sensor_pos_y = static_cast<double>(m_sensorSpec.posY);
	const double sensor_pos_z = static_cast<double>(m_sensorSpec.posZ);
	const double sensor_rot_yaw_deg = static_cast<double>(m_sensorSpec.yawDeg);
	const double sensor_rot_pitch_deg = static_cast<double>(m_sensorSpec.pitchDeg);
	const double sensor_range = static_cast<double>(m_sensorSpec.range);

	const double tan_half_hfov = std::tan(hfov_deg * 0.5 * pi / 180.0);
	const double tan_half_vfov = std::tan(vfov_deg * 0.5 * pi / 180.0);

	const double yaw_rad = sensor_rot_yaw_deg * pi / 180.0;
	const double pitch_rad = sensor_rot_pitch_deg * pi / 180.0;
	const double cos_yaw = std::cos(yaw_rad);
	const double sin_yaw = std::sin(yaw_rad);
	const double cos_pitch = std::cos(pitch_rad);
	const double sin_pitch = std::sin(pitch_rad);

	auto transform_to_world = [&](double lx, double ly, double lz) -> ImPlot3DPoint {
		double y_pitch = ly * cos_pitch - lz * sin_pitch;
		double z_pitch = ly * sin_pitch + lz * cos_pitch;

		double x_yaw = lx * cos_yaw - y_pitch * sin_yaw;
		double y_yaw = lx * sin_yaw + y_pitch * cos_yaw;

		return ImPlot3DPoint(x_yaw + sensor_pos_x, y_yaw + sensor_pos_y, z_pitch + sensor_pos_z);
	};

	const double near_hw = sensor_width * 0.5;
	const double near_hh = sensor_height * 0.5;
	std::array<ImPlot3DPoint, 4> near_corners = {
		transform_to_world(-near_hw, 0.0, -near_hh),
		transform_to_world( near_hw, 0.0, -near_hh),
		transform_to_world( near_hw, 0.0,  near_hh),
		transform_to_world(-near_hw, 0.0,  near_hh)
	};

	const double far_hw = near_hw + sensor_range * tan_half_hfov;
	const double far_hh = near_hh + sensor_range * tan_half_vfov;
	std::array<ImPlot3DPoint, 4> far_corners = {
		transform_to_world(-far_hw, sensor_range, -far_hh),
		transform_to_world( far_hw, sensor_range, -far_hh),
		transform_to_world( far_hw, sensor_range,  far_hh),
		transform_to_world(-far_hw, sensor_range,  far_hh)
	};

	std::array<double, 24> line_x, line_y, line_z;
	size_t line_idx = 0;

	auto add_line = [&](const ImPlot3DPoint& p1, const ImPlot3DPoint& p2) {
		line_x[line_idx] = p1.x; line_y[line_idx] = p1.y; line_z[line_idx] = p1.z; line_idx++;
		line_x[line_idx] = p2.x; line_y[line_idx] = p2.y; line_z[line_idx] = p2.z; line_idx++;
	};

	for (int i = 0; i < 4; ++i) {
		int next = (i + 1) % 4;
		add_line(near_corners[i], near_corners[next]);
		add_line(far_corners[i], far_corners[next]);
		add_line(near_corners[i], far_corners[i]);
	}

	ImPlot3DSpec line_spec;
	line_spec.LineColor = ImVec4(0.7f, 0.9f, 1.0f, polygon ? 0.9f : 0.6f);
	line_spec.LineWeight = polygon ? 2.0f : 1.5f;
	line_spec.Marker = ImPlot3DMarker_None;
	line_spec.Flags = ImPlot3DItemFlags_NoLegend | ImPlot3DLineFlags_Segments;

	ImPlot3D::PlotLine("##FrustumOutline", line_x.data(), line_y.data(), line_z.data(), 24, line_spec);

	if (polygon)
	{
		std::array<double, 36> tri_x, tri_y, tri_z;
		size_t tri_idx = 0;

		auto add_triangle = [&](int i1, int i2, int i3) {
			const ImPlot3DPoint* pts[3] = {
				i1 < 4 ? &near_corners[i1] : &far_corners[i1 - 4],
				i2 < 4 ? &near_corners[i2] : &far_corners[i2 - 4],
				i3 < 4 ? &near_corners[i3] : &far_corners[i3 - 4]
			};
			for (int j = 0; j < 3; ++j) {
				tri_x[tri_idx] = pts[j]->x;
				tri_y[tri_idx] = pts[j]->y;
				tri_z[tri_idx] = pts[j]->z;
				tri_idx++;
			}
		};

		add_triangle(0, 1, 2); add_triangle(0, 2, 3); // Near
		add_triangle(4, 5, 6); add_triangle(4, 6, 7); // Far
		add_triangle(0, 3, 7); add_triangle(0, 7, 4); // Left
		add_triangle(1, 2, 6); add_triangle(1, 6, 5); // Right
		add_triangle(3, 2, 6); add_triangle(3, 6, 7); // Top
		add_triangle(0, 1, 5); add_triangle(0, 5, 4); // Bottom

		ImPlot3DSpec poly_spec;
		poly_spec.FillColor = ImVec4(0.7f, 0.9f, 1.0f, 0.15f);
		poly_spec.LineColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		poly_spec.Marker = ImPlot3DMarker_None;
		poly_spec.Flags = ImPlot3DItemFlags_NoLegend;

		ImPlot3D::PlotTriangle("##FrustumVolume", tri_x.data(), tri_y.data(), tri_z.data(), 36, poly_spec);
	}
}

void PointCloudWindow::drawCenteredText(const char* text)
{
	auto& plot_style = ImPlot3D::GetStyle();

	const ImVec2 padding = plot_style.PlotPadding;
	const ImVec2 window_pos = ImGui::GetWindowPos();
	const ImVec2 window_size = ImGui::GetWindowSize();

	const ImVec2 p_min = ImVec2(window_pos.x + padding.x, window_pos.y + padding.y);
	const ImVec2 p_max = ImVec2(window_pos.x + window_size.x - padding.x, window_pos.y + window_size.y - padding.y);

	const ImVec2 text_padding = ImVec2(20.f, 20.f);
	const ImVec2 text_size = ImGui::CalcTextSize(text);
	const ImVec2 center = ImVec2((p_min.x + p_max.x) * 0.5f, (p_min.y + p_max.y) * 0.5f);
	const ImVec2 text_pos = ImVec2(center.x - text_size.x * 0.5f, center.y - text_size.y * 0.5f);
	const ImVec2 text_bg_min = ImVec2(text_pos.x - text_padding.x, text_pos.y - text_padding.y);
	const ImVec2 text_bg_max = ImVec2(text_pos.x + text_size.x + text_padding.x, text_pos.y + text_size.y + text_padding.y);
	
	const ImU32 text_bg_color = ImGui::GetColorU32(plot_style.Colors[ImPlot3DCol_PlotBg]);
	const ImU32 text_color = ImGui::GetColorU32(ImGuiCol_Text);

	auto* draw_list = ImGui::GetWindowDrawList();
	draw_list->AddRectFilled(text_bg_min, text_bg_max, text_bg_color);
	draw_list->AddText(text_pos, text_color, text);
}

void PointCloudWindow::resetPlaybackState()
{
	m_playbackTimer = 0.0;
	m_playbackAccumulated = 0.0;
}

void PointCloudWindow::updatePlaybackState()
{
	if (!m_isPlaying || m_currSession == nullptr || m_currSession->frames.empty())
		return;

	if (m_playbackIdx >= m_currSession->frames.size()) {
		m_playbackIdx = m_currSession->frames.size() - 1;
		m_isPlaying = false;
		return;
	}

	m_playbackTimer += m_playbackSpeed * ImGui::GetIO().DeltaTime;

	while (m_isPlaying && m_playbackIdx < m_currSession->frames.size() - 1)
	{
		double delay = m_currSession->frames[m_playbackIdx + 1].frame.deltaUs / 1e6;

		// calibrate delay to avoid extreme values due to timestamp issues
		if (delay > 0.2) delay = 0.05;
		if (delay <= 0.0) delay = 0.001;

		if (m_playbackAccumulated + delay < m_playbackTimer)
		{
			m_playbackAccumulated += delay;
			m_playbackIdx++;
		}
		else
		{
			break;
		}
	}

	if (m_playbackIdx >= m_currSession->frames.size() - 1)
	{
		m_playbackIdx = m_currSession->frames.size() - 1;
		m_isPlaying = false;
	}
}

ImU32 PointCloudWindow::getPointColor(const float doppler, float power) const
{
	if (m_visualSetting.allWhite)
		return 0xFFFFFFFF;

	const float doppler_span = m_visualSetting.dopplerMax - m_visualSetting.dopplerMin;
	const float power_span = m_visualSetting.powerDBMax - m_visualSetting.powerDBMin;
	const float color_t = doppler_span > 0.0f
		? std::clamp((doppler - m_visualSetting.dopplerMin) / doppler_span, 0.0f, 1.0f)
		: 0.0f;

	const float power_db = 10.0f * std::log10(power + 1.0f);
	const float alpha_t = (power_db - m_visualSetting.powerDBMin) / power_span;
	float alpha;

	if (m_visualSetting.powerCutoffEnabled)
	{
		alpha = alpha_t <= 0.0f ? 0.0f : std::clamp(alpha_t, 0.0f, 1.0f);
	}
	else
	{
		alpha = std::clamp(alpha_t, 0.2f, 1.0f);
	}

	const ImVec4 color(1.0f - color_t, 0.0f, color_t, alpha);
	
	return ImGui::ColorConvertFloat4ToU32(color);
}

ImU64 PointCloudWindow::makeScatterContentKey() const
{
	ImU64 hash = 14695981039346656037ull;
	hash = fnv1a_value(hash, m_visualSetting.dopplerMin);
	hash = fnv1a_value(hash, m_visualSetting.dopplerMax);
	hash = fnv1a_value(hash, m_visualSetting.powerDBMin);
	hash = fnv1a_value(hash, m_visualSetting.powerDBMax);
	hash = fnv1a_value(hash, m_visualSetting.powerCutoffEnabled);
	hash = fnv1a_value(hash, m_visualSetting.pointSize);
	hash = fnv1a_value(hash, m_visualSetting.allWhite);
	hash = fnv1a_value(hash, m_sessionId);
	hash = fnv1a_value(hash, m_playbackIdx);

	if (m_currFrame == nullptr)
		return hash == 0 ? 1 : hash;

	hash = fnv1a_value(hash, m_currFrame->frameCount);
	hash = fnv1a_value(hash, m_currFrame->packetSize);
	hash = fnv1a_value(hash, m_currFrame->deltaUs);
	const size_t point_count = m_currFrame->points.size();
	hash = fnv1a_value(hash, point_count);
	if (point_count > 0)
		hash = fnv1a(hash, &m_currFrame->points.front(), sizeof(retina::Point));
	if (point_count > 1)
		hash = fnv1a(hash, &m_currFrame->points.back(), sizeof(retina::Point));
	if (point_count > 2)
		hash = fnv1a(hash, &m_currFrame->points[point_count / 2], sizeof(retina::Point));

	return hash == 0 ? 1 : hash;
}
