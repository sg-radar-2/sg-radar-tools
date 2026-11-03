#pragma once

#include "recording_window.h"

#include "../../device/retina.h"

#include <implot3d.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class PointCloudWindow : public DockingWindow
{
	friend class AnalysisWindow;

public:
	enum class WindowSource
	{
		None = 0,
		Sensor = 1,
		CurrentSession = 2,
		Session = 3,
	};

	struct VisualizeSetting
	{
		float dopplerMin = -0.5f;
		float dopplerMax = 0.5f;

		float powerDBMin = 30.0f;
		float powerDBMax = 53.0f;
		bool powerCutoffEnabled = false;

		float pointSize = 2.0f;
		bool allWhite = false;
	};

	PointCloudWindow(const std::string& title);

	void setRecordingWindow(RecordingWindow* window);
	void setSensorSpec(const retina::SensorSpec& spec);

	const retina::Frame& getLastFrame() const;

private:
	void onUpdate() override;
	void onDraw(const ImVec2& size) override;
	void onDrawMenuBar();

	void onPlayButtonClicked();
	void onRecordButtonClicked();
	void onTrimLeftButtonClicked();
	void onTrimRightButtonClicked();

	void onSettingReadLine(void* entry, const char* line) override;
	void onSettingWriteAll(ImGuiTextBuffer* buf) const override;

	void drawOrientationLabel();
	void drawPlaybackControls();
	void drawFrustum(bool polygon = false);
	void drawTargetBox(const retina::Target& target, const ImPlot3DSpec& spec);
	bool queueTargetPictogram(const retina::Target& target);
	void drawQueuedPictograms();
	void drawCenteredText(const char* text);

	void resetPlaybackState();
	void updatePlaybackState();

	ImU32 getPointColor(const float doppler, float power) const;
	ImU64 makeScatterContentKey() const;

	RecordingWindow* m_recordingWindow = nullptr;

	VisualizeSetting m_visualSetting;
	retina::SensorSpec m_sensorSpec;

	std::string m_plotTitle;
	std::string m_plotLabel;
	std::string m_gizmoLabel;
	std::string m_sliderLabel;

	ImPlot3DQuat m_viewRotation;
	ImPlot3DBox m_viewAxesBox;
	double m_viewZoom = 1.0;
	WindowSource m_source = WindowSource::None;

	SessionID m_sessionId = static_cast<SessionID>(-1);
	const PointCloudRecorder::RecordSession* m_currSession = nullptr;
	const retina::Frame* m_currFrame = nullptr;

	int m_playbackIdx = 0;
	int m_speedIdx = 0;
	double m_playbackSpeed = 1.0;
	double m_playbackTimer = 0.0;
	double m_playbackAccumulated = 0.0;
	bool m_isPlaying = false;

	int m_recordMaxFrameLength = INT32_MAX;
	bool m_isRecording = false;

	bool m_showTargets = false;
	bool m_showTargetPictogram = false;

	// Pictograms are drawn after EndPlot() so they overlay the 3D layer instead of sitting under it.
	struct QueuedPictogram
	{
		ImVec2 center;
		unsigned int texture;
		uint32_t targetId;
	};
	std::vector<QueuedPictogram> m_queuedPictograms;
	ImVec2 m_pictogramClipMin;
	ImVec2 m_pictogramClipMax;

	// Display-only smoothing of the pictogram height (the packet has no z, so it comes from the noisy point box).
	struct SmoothedHeight
	{
		double z;
		int lastFrame;
	};
	std::unordered_map<uint32_t, SmoothedHeight> m_pictogramHeights;
	bool m_showGizmo = true;
	int m_frustumType = 0;
};
