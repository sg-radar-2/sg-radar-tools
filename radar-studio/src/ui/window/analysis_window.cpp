#include "analysis_window.h"
#include "../../my_app.h"

#include <common/ui/imgui_custom.h>

#include <implot.h>

#include <algorithm>
#include <cmath>

AnalysisWindow::AnalysisWindow() :
	DockingWindow("Analysis", true),
	m_source(SourceType::None),
	m_sourceRecordingWindow(nullptr)
{
	m_dopplerHistogramX.resize(kHistogramBins, 0.0f);
	m_dopplerHistogramCount.resize(kHistogramBins, 0);
	m_powerHistogramX.resize(kHistogramBins, 0.0f);
	m_powerHistogramCount.resize(kHistogramBins, 0);
}

void AnalysisWindow::setSourceRecordingWindow(RecordingWindow* window)
{
	m_sourceRecordingWindow = window;
}

void AnalysisWindow::setSourcePointWindows(const std::vector<PointCloudWindow*>& windows)
{
	m_sourcePointWindows = windows;
}

void AnalysisWindow::onUpdate()
{

}

void AnalysisWindow::onDraw(const ImVec2& size)
{
	drawSourceSelector();

	ImGui::Spacing();
	ImGui::Separator();

	if (m_source == SourceType::None)
	{
		m_currSession = nullptr;
		m_sessionId = -1;
		m_currFrame = nullptr;
		m_playbackIdx = 0;

		ImGui::TextUnformatted("Please select a data source");
		return;
	}

	if (m_source == SourceType::Sensor)
	{
		if (MyApp::get().getConnectionStatus() != MyApp::ConnectionStatus::Connected)
		{
			ImGui::TextUnformatted("Sensor is not connected");
			return;
		}
	}
	else if (m_source == SourceType::CurrentSession || m_source == SourceType::Session)
	{
		if (m_sourceRecordingWindow)
		{
			if (m_source == SourceType::CurrentSession)
			{
				m_currSession = m_sourceRecordingWindow->getCurrentSession();
				m_sessionId = m_currSession ? m_currSession->id : -1;
			}
			else
			{
				drawSessionSelector();
			}
		}
		else
		{
			ImGui::TextUnformatted("No recording window available");
			return;
		}
	}
	else if (m_source >= SourceType::Window1 && m_source <= SourceType::Window4)
	{
		const auto window_idx = static_cast<int>(m_source) - static_cast<int>(SourceType::Window1);
		const auto* point_window = window_idx >= 0 && static_cast<size_t>(window_idx) < m_sourcePointWindows.size() ? m_sourcePointWindows[window_idx] : nullptr;

		if (point_window)
		{
			const auto window_source = point_window->m_source;

			if (window_source == PointCloudWindow::WindowSource::None)
			{
				ImGui::TextUnformatted("The selected window has no data source");
				return;
			}
			else if (window_source == PointCloudWindow::WindowSource::Sensor)
			{
				if (MyApp::get().getConnectionStatus() != MyApp::ConnectionStatus::Connected)
				{
					ImGui::TextUnformatted("Sensor is not connected");
					return;
				}
			}
			else if (window_source != PointCloudWindow::WindowSource::None)
			{
				m_sessionId = point_window->m_sessionId;
				m_currSession = point_window->m_currSession;
				m_currFrame = point_window->m_currFrame;
				m_playbackIdx = point_window->m_playbackIdx;
			}
		}
		else
		{
			ImGui::TextUnformatted("No point cloud window available");
			return;
		}
	}

	syncCurrentFrameFromSource();
	updateSelectedTargetsFromFrame();

	if (m_source == SourceType::CurrentSession || m_source == SourceType::Session)
		drawFrameSelector();

	drawTargetSelector();
	drawHistograms();
	drawFrameInfo();
}

void AnalysisWindow::onSettingReadLine(void* entry, const char* line)
{
	int source_idx;
	if (sscanf(line, "Source=%d", &source_idx) == 1)
		m_source = static_cast<SourceType>(source_idx);
	else if (sscanf(line, "SessionID=%d", &source_idx) == 1)
		m_sessionId = source_idx;
	else if (sscanf(line, "FrameIndex=%d", &source_idx) == 1)
		m_playbackIdx = source_idx;
}

void AnalysisWindow::onSettingWriteAll(ImGuiTextBuffer* buf) const
{
	buf->appendf("Source=%d\n", static_cast<int>(m_source));
	buf->appendf("SessionID=%d\n", m_sessionId);
	buf->appendf("FrameIndex=%d\n", m_playbackIdx);
}

void AnalysisWindow::drawSourceSelector()
{
	ImGui::TextUnformatted("Data Source:");
	ImGui::SameLine();

	const char* source_items = "None\0Sensor\0Current Session\0Session\0Window 1\0Window 2\0Window 3\0Window 4\0";

	ImGui::SetNextItemWidth(200.0f);
	if (ImGui::Combo("##AnalysisSource", &reinterpret_cast<int&>(m_source), source_items))
	{
		m_playbackIdx = 0;
		m_sessionId = -1;
		m_selectedTargets.clear();
		m_currSession = nullptr;
		m_currFrame = nullptr;
		m_histogramDirty = true;
	}
}

void AnalysisWindow::drawSessionSelector()
{
	const auto& recorder = m_sourceRecordingWindow->getRecorder();

	if (recorder.getSessionCount() == 0)
	{
		ImGui::TextUnformatted("No sessions available");
		return;
	}

	const auto* selected_item = recorder.getSession(m_sessionId);
	const char* priview_text = selected_item ? selected_item->name.c_str() : "Select Session";
	const int session_count = recorder.getSessionCount();

	ImGui::SetNextItemWidth(-FLT_MAX);
	if (ImGui::BeginCombo("##AnalysisSession", priview_text))
	{
		for (int i = 0; i < session_count; ++i)
		{
			const auto id = recorder.getOrderedSessionID(i);
			const auto& item = recorder.getSession(id);

			if (ImGui::Selectable(item->name.c_str(), id == m_sessionId))
			{
				m_sessionId = id;
				m_playbackIdx = 0;
				m_histogramDirty = true;
				m_currSession = recorder.getSession(m_sessionId);
				m_currFrame = (m_currSession && !m_currSession->frames.empty()) ? &m_currSession->frames[0].frame : nullptr;
				updateSelectedTargetsFromFrame();
			}
		}
		ImGui::EndCombo();
	}
}

void AnalysisWindow::drawFrameSelector()
{
	const int frame_count = (m_source == SourceType::CurrentSession || m_source == SourceType::Session) && m_currSession
		? static_cast<int>(m_currSession->frames.size())
		: 1;

	if (frame_count <= 1)
		return;

	ImGui::TextUnformatted("Frame:");
	ImGui::SameLine();

	ImGui::SetNextItemWidth(100.0f);
	if (ImGui::SliderInt("##FrameSlider", &m_playbackIdx, 0, frame_count - 1))
	{
		m_histogramDirty = true;
		syncCurrentFrameFromSource();
		updateSelectedTargetsFromFrame();
	}

	ImGui::SameLine();
	char frame_label[64];
	snprintf(frame_label, sizeof(frame_label), "%d / %d", m_playbackIdx, frame_count - 1);
	ImGui::TextUnformatted(frame_label);
}

void AnalysisWindow::drawFrameInfo()
{
	if (!m_currFrame)
		return;

	ImGui::SeparatorText("Frame Information:");
	ImGui::Text("Frame Count: %u", m_currFrame->frameCount);
	ImGui::Text("Packet Size: %u bytes", m_currFrame->packetSize);
	ImGui::Text("Delta Time : %.3f ms", m_currFrame->deltaUs / 1000.0);
	ImGui::Text("Points     : %zu", m_currFrame->points.size());
	ImGui::Text("Targets    : %zu", m_currFrame->targets.size());
}

void AnalysisWindow::drawTargetSelector()
{
	ImGui::SeparatorText("Target Selection");

	if (!m_currFrame)
	{
		ImGui::TextUnformatted("Empty frame");
		return;
	}

	if (ImGui::Checkbox("select all", &m_selectAllTargets))
	{
		for (auto& selected : m_selectedTargets)
			selected = m_selectAllTargets;
		m_histogramDirty = true;
	}

	// Target check list
	const auto child_flags = ImGuiChildFlags_Borders;

	ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
	if (ImGui::BeginChild("##TargetListChild", ImVec2(0, 100), child_flags))
	{
		bool selected_all = true;

		for (const auto& target : m_currFrame->targets)
		{
			char label[128];
			bool selected = target.targetId < m_selectedTargets.size() ? m_selectedTargets[target.targetId] != 0 : false;

			snprintf(label, sizeof(label), "ID %u(x=%.2f, y=%.2f)", target.targetId, target.x, target.y);
			if (ImGui::Selectable(label, &selected))
			{
				if (target.targetId >= m_selectedTargets.size())
					m_selectedTargets.resize(target.targetId + 1, 0);
				m_selectedTargets[target.targetId] = selected ? 1 : 0;
				m_histogramDirty = true;
			}

			selected_all = selected_all && selected;
		}
	}
	ImGui::EndChild();

	ImGui::Dummy(ImVec2(0.0f, 15.0f));
}

void AnalysisWindow::drawHistograms()
{
	ImGui::SeparatorText("Histograms");

	if (!m_currFrame)
	{
		std::fill(m_dopplerHistogramCount.begin(), m_dopplerHistogramCount.end(), 0.0f);
		std::fill(m_powerHistogramCount.begin(), m_powerHistogramCount.end(), 0.0f);
		m_dopplerMaxCount = 0.0f;
		m_powerMaxCount = 0.0f;
		m_histogramDirty = false;
		ImGui::TextUnformatted("No point cloud data");
		return;
	}

	if (m_histogramDirty)
	{
		std::vector<float> doppler_data;
		std::vector<float> power_data;

		bool has_selection = std::any_of(m_selectedTargets.begin(), m_selectedTargets.end(), [](uint8_t b) { return b != 0; });

		for (const auto& point : m_currFrame->points)
		{
			if (has_selection && point.targetId >= 0 && point.targetId < static_cast<int>(m_currFrame->targets.size()))
			{
				bool is_selected = false;
				for (size_t i = 0; i < m_currFrame->targets.size(); ++i)
				{
					if (m_currFrame->targets[i].targetId == point.targetId && i < m_selectedTargets.size())
					{
						is_selected = m_selectedTargets[i] != 0;
						break;
					}
				}
				if (!is_selected)
					continue;
			}

			doppler_data.push_back(point.doppler);
			power_data.push_back(point.power);
		}

		buildDopplerHistogram(doppler_data, m_dopplerHistogramCount);
		buildPowerHistogram(power_data, m_powerHistogramCount);
		m_histogramDirty = false;
	}

	const auto plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoTitle | ImPlotFlags_NoInputs;
	const float plot_width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
	const ImVec2 plot_size(plot_width, 200.0f);
	const ImGuiIO& io = ImGui::GetIO();
	const float wheel_zoom_factor = std::pow(0.90f, io.MouseWheel);

	auto applyHistogramWheel = [&](bool doppler)
	{
		if (io.MouseWheel == 0.0f)
			return;

		if (doppler)
		{
			if (io.KeyCtrl)
			{
				const float half_range = std::max(0.01f, (m_analysisSetting.dopplerMax - m_analysisSetting.dopplerMin) * 0.5f);
				const float next_half_range = std::clamp(half_range * wheel_zoom_factor, 0.01f, 10.0f);
				if (next_half_range != half_range)
				{
					m_analysisSetting.dopplerMin = -next_half_range;
					m_analysisSetting.dopplerMax = next_half_range;
					m_histogramDirty = true;
				}
			}
			else
			{
				m_dopplerHeightScale = std::clamp(m_dopplerHeightScale * wheel_zoom_factor, 0.1f, 100.0f);
			}
		}
		else
		{
			if (io.KeyCtrl)
			{
				const float range = std::max(0.01f, m_analysisSetting.powerMax - m_analysisSetting.powerMin);
				const float next_range = std::clamp(range * wheel_zoom_factor, 0.01f, 1000000.0f);
				const float next_max = m_analysisSetting.powerMin + next_range;
				if (next_max != m_analysisSetting.powerMax)
				{
					m_analysisSetting.powerMax = next_max;
					m_histogramDirty = true;
				}
			}
			else
			{
				m_powerHeightScale = std::clamp(m_powerHeightScale * wheel_zoom_factor, 0.1f, 100.0f);
			}
		}
	};

	ImPlotSpec spec;
	spec.FillColor = ImVec4(0.8f, 0.8f, 0.8f, 0.9f);

	// Doppler histogram
	ImGui::TextUnformatted("Doppler Distribution");
	if (ImPlot::BeginPlot("##DopplerHistogram", plot_size, plot_flags))
	{
		ImPlot::SetupAxes(nullptr, nullptr, 0, ImPlotAxisFlags_NoDecorations);
		ImPlot::SetupAxisLimits(ImAxis_X1, m_analysisSetting.dopplerMin, m_analysisSetting.dopplerMax, ImPlotCond_Always);
		ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(1.0f, m_dopplerMaxCount * m_dopplerHeightScale), ImPlotCond_Always);

		const float step = (m_analysisSetting.dopplerMax - m_analysisSetting.dopplerMin) / kHistogramBins;
		m_dopplerHistogramX.resize(kHistogramBins);
		for (int i = 0; i < kHistogramBins; ++i)
			m_dopplerHistogramX[i] = m_analysisSetting.dopplerMin + step * (i + 0.5f);

		ImPlot::PlotBars("Doppler", m_dopplerHistogramX.data(), m_dopplerHistogramCount.data(), kHistogramBins, step * 0.9f, spec);
		if (ImPlot::IsPlotHovered())
			applyHistogramWheel(true);
		ImPlot::EndPlot();
	}

	MyGui::LegendBar(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
	ImGui::SetNextItemWidth(plot_width);
	ImGui::DragFloatRange2("##DopplerRange", &m_analysisSetting.dopplerMin, &m_analysisSetting.dopplerMax,
		0.01f, -10.0f, 10.0f, "Min: %.2f", "Max: %.2f");
	if (ImGui::IsItemEdited())
		m_histogramDirty = true;

	// Power histogram
	ImGui::Spacing();
	ImGui::TextUnformatted("Power Distribution");
	if (ImPlot::BeginPlot("##PowerHistogram", plot_size, plot_flags))
	{
		ImPlot::SetupAxes(nullptr, nullptr, 0, ImPlotAxisFlags_NoDecorations);
		ImPlot::SetupAxisLimits(ImAxis_X1, m_analysisSetting.powerMin, m_analysisSetting.powerMax, ImPlotCond_Always);
		ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(5.0f, m_powerMaxCount * m_powerHeightScale), ImPlotCond_Always);

		const float step = (m_analysisSetting.powerMax - m_analysisSetting.powerMin) / kHistogramBins;
		m_powerHistogramX.resize(kHistogramBins);
		for (int i = 0; i < kHistogramBins; ++i)
		{
			m_powerHistogramX[i] = m_analysisSetting.powerMin + step * (i + 0.5f);
		}

		ImPlot::PlotBars("Power", m_powerHistogramX.data(), m_powerHistogramCount.data(), kHistogramBins, step * 0.9f, spec);
		if (ImPlot::IsPlotHovered())
			applyHistogramWheel(false);
		ImPlot::EndPlot();
	}

	MyGui::LegendBar(ImVec4(0.0f, 0.0f, 0.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
	ImGui::SetNextItemWidth(plot_width);
	ImGui::DragFloatRange2("##PowerRange", &m_analysisSetting.powerMin, &m_analysisSetting.powerMax,
		1.0f, 0.0f, 120.0f, "Min: %.0f", "Max: %.0f");
	if (ImGui::IsItemEdited())
		m_histogramDirty = true;

	ImGui::Dummy(ImVec2(0.0f, 15.0f));
}

void AnalysisWindow::syncCurrentFrameFromSource()
{
	const auto previous_source = m_lastHistogramSource;
	const auto previous_session_id = m_lastHistogramSessionId;
	const auto previous_playback_idx = m_lastHistogramPlaybackIdx;
	const auto previous_frame_count = m_lastHistogramFrameCount;
	const auto* previous_frame = m_currFrame;
	const auto* previous_session = m_currSession;

	const retina::Frame* next_frame = nullptr;
	const PointCloudRecorder::RecordSession* next_session = nullptr;
	SessionID next_session_id = -1;
	int next_playback_idx = 0;

	if (m_source == SourceType::Sensor)
	{
		if (MyApp::get().getConnectionStatus() == MyApp::ConnectionStatus::Connected)
			next_frame = &MyApp::get().getLastFrame();
	}
	else if (m_source == SourceType::CurrentSession || m_source == SourceType::Session)
	{
		if (m_sourceRecordingWindow)
		{
			if (m_source == SourceType::CurrentSession)
				next_session = m_sourceRecordingWindow->getCurrentSession();
			else
				next_session = m_sourceRecordingWindow->getRecorder().getSession(m_sessionId);

			if (next_session != nullptr)
			{
				next_session_id = next_session->id;
				if (!next_session->frames.empty())
				{
					next_playback_idx = std::clamp(m_playbackIdx, 0, static_cast<int>(next_session->frames.size()) - 1);
					next_frame = &next_session->frames[next_playback_idx].frame;
				}
			}
		}
	}
	else if (m_source >= SourceType::Window1 && m_source <= SourceType::Window4)
	{
		const auto window_idx = static_cast<size_t>(static_cast<int>(m_source) - static_cast<int>(SourceType::Window1));
		if (window_idx < m_sourcePointWindows.size())
		{
			const auto* point_window = m_sourcePointWindows[window_idx];
			if (point_window != nullptr)
			{
				if (point_window->m_source == PointCloudWindow::WindowSource::Sensor)
				{
					if (MyApp::get().getConnectionStatus() == MyApp::ConnectionStatus::Connected)
						next_frame = &MyApp::get().getLastFrame();
				}
				else if (point_window->m_source != PointCloudWindow::WindowSource::None)
				{
					next_session_id = point_window->m_sessionId;
					next_session = point_window->m_currSession;
					next_frame = point_window->m_currFrame;
					next_playback_idx = point_window->m_playbackIdx;
				}
			}
		}
	}

	m_currSession = next_session;
	m_sessionId = next_session_id;
	m_playbackIdx = next_playback_idx;
	m_currFrame = next_frame;

	const bool frame_changed = previous_frame != m_currFrame || previous_session != m_currSession || previous_source != m_source || previous_session_id != m_sessionId || previous_playback_idx != m_playbackIdx || (m_currFrame && m_currFrame->frameCount != previous_frame_count);
	if (frame_changed)
		m_histogramDirty = true;

	m_lastHistogramSource = m_source;
	m_lastHistogramSessionId = m_sessionId;
	m_lastHistogramPlaybackIdx = m_playbackIdx;
	m_lastHistogramFrameCount = m_currFrame ? m_currFrame->frameCount : 0;
}

void AnalysisWindow::updateSelectedTargetsFromFrame()
{
	if (!m_currFrame)
	{
		m_selectedTargets.clear();
		return;
	}

	bool selected_all = true;

	for (const auto& target : m_currFrame->targets)
	{
		if (target.targetId >= m_selectedTargets.size())
			m_selectedTargets.resize(target.targetId + 1, 0);
		if (m_selectedTargets[target.targetId] == 0)
			selected_all = false;
	}

	m_selectAllTargets = selected_all;
}

void AnalysisWindow::buildDopplerHistogram(const std::vector<float>& data, std::vector<float>& histogram)
{
	std::fill(histogram.begin(), histogram.end(), 0);
	m_dopplerMaxCount = 0.0f;

	if (data.empty())
		return;

	float min_val = m_analysisSetting.dopplerMin;
	float max_val = m_analysisSetting.dopplerMax;
	float range = max_val - min_val;
	if (range <= 0.0f)
		range = 1.0f;

	for (float val : data)
	{
		if (val >= min_val && val <= max_val)
		{
			int bin = static_cast<int>((val - min_val) / range * kHistogramBins);
			bin = std::clamp(bin, 0, kHistogramBins - 1);
			histogram[bin]++;
			m_dopplerMaxCount = std::max(m_dopplerMaxCount, histogram[bin]);
		}
	}
}

void AnalysisWindow::buildPowerHistogram(const std::vector<float>& data, std::vector<float>& histogram)
{
	std::fill(histogram.begin(), histogram.end(), 0);
	m_powerMaxCount = 0.0f;

	if (data.empty())
		return;

	float min_val = m_analysisSetting.powerMin;
	float max_val = m_analysisSetting.powerMax;
	float range = max_val - min_val;
	if (range <= 0.0f)
		range = 1.0f;

	for (float val : data)
	{
		float db = 10.0f * std::log10(val + 1.0f);

		if (db >= min_val && db <= max_val)
		{
			int bin = static_cast<int>((db - min_val) / range * kHistogramBins);
			bin = std::clamp(bin, 0, kHistogramBins - 1);
			histogram[bin]++;
			m_powerMaxCount = std::max(m_powerMaxCount, histogram[bin]);
		}
	}
}
