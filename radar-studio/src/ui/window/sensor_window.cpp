#include "sensor_window.h"
#include "../../my_app.h"

#include <cstring>

namespace
{
	constexpr int kSpecFieldCount = 3;

	const char* kSpecLabels[] = {
		"fov H:",
		"fov V:",
		"Size:",
		"Position:",
		"Rotation:",
		"Range X:",
		"Range Y:",
		"Range Z:",
		"Range:",
	};

	bool sensorSpecEqual(const retina::SensorSpec& a, const retina::SensorSpec& b)
	{
		return std::memcmp(&a, &b, sizeof(retina::SensorSpec)) == 0;
	}

	void drawDirtyMark(bool dirty)
	{
		if (!dirty)
			return;

		ImGui::SameLine(0.0f, 4.0f);
		ImGui::TextUnformatted("*");
	}

	float specLabelColumnWidth()
	{
		float width = 0.0f;
		for (const char* label : kSpecLabels)
			width = ImMax(width, ImGui::CalcTextSize(label).x);
		return width;
	}

	float specFieldWidth(float label_width)
	{
		const ImGuiStyle& style = ImGui::GetStyle();
		const float star_reserve = ImGui::CalcTextSize("*").x + style.ItemSpacing.x;
		const float avail = ImGui::GetContentRegionAvail().x - label_width - style.ItemInnerSpacing.x - star_reserve;
		return ImMax(50.0f, (avail - style.ItemInnerSpacing.x * (kSpecFieldCount - 1)) / static_cast<float>(kSpecFieldCount));
	}

	float specVecWidth(float field_width, int components)
	{
		return field_width * static_cast<float>(components)
			+ ImGui::GetStyle().ItemInnerSpacing.x * static_cast<float>(components - 1);
	}

	void beginSpecRow(const char* label, float label_width)
	{
		const float start_x = ImGui::GetCursorPosX();
		ImGui::AlignTextToFramePadding();
		ImGui::SetCursorPosX(start_x + label_width - ImGui::CalcTextSize(label).x);
		ImGui::TextUnformatted(label);
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
	}

	void drawMinMaxRow(const char* label, float label_width, float field_width,
		const char* min_id, const char* max_id, float* min_value, float* max_value, bool dirty)
	{
		const float inner = ImGui::GetStyle().ItemInnerSpacing.x;

		beginSpecRow(label, label_width);
		ImGui::SetNextItemWidth(field_width);
		ImGui::InputFloat(min_id, min_value, 0.0f, 0.0f, "%.2f");
		ImGui::SameLine(0.0f, inner);
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted("~");
		ImGui::SameLine(0.0f, inner);
		ImGui::SetNextItemWidth(field_width);
		ImGui::InputFloat(max_id, max_value, 0.0f, 0.0f, "%.2f");
		drawDirtyMark(dirty);
	}
}

SensorWindow::SensorWindow() :
	DockingWindow("Sensor"),
	m_draftSensorSpec(MyApp::get().getDeviceInfo().sensorSpec)
{
}

void SensorWindow::syncDraftFromDevice()
{
	m_draftSensorSpec = MyApp::get().getDeviceInfo().sensorSpec;
}

void SensorWindow::onDraw(const ImVec2& size)
{
	(void)size;

	drawConnection();

	ImGui::Spacing();
	ImGui::SeparatorText("Sensor Info");
	drawSensorInfo();

	ImGui::Spacing();
	ImGui::SeparatorText("Sensor spec");
	drawSensorSpec();
}

void SensorWindow::drawConnection()
{
	auto& app = MyApp::get();
	const auto status = app.getConnectionStatus();
	const auto& network_info = app.getLocalNetworkInfo();
	const auto& device_info = app.getDeviceInfo();

	ImGui::SeparatorText("Connection");
	ImGui::BeginDisabled(network_info.address.empty());
	ImGui::Text("Gateway   : %s", network_info.address.empty() ? "N/A" : network_info.address.c_str());
	ImGui::Text("Subnetmask: %s", network_info.subnetMask.empty() ? "N/A" : network_info.subnetMask.c_str());
	ImGui::EndDisabled();
	ImGui::BeginDisabled(status != MyApp::ConnectionStatus::Connected);
	ImGui::Text("Device IP : %s", device_info.ip.empty() ? "N/A" : device_info.ip.c_str());
	ImGui::Text("Device MAC: %s", device_info.mac.empty() ? "N/A" : device_info.mac.c_str());
	ImGui::EndDisabled();

	if (status == MyApp::ConnectionStatus::Disconnected || status == MyApp::ConnectionStatus::Failed)
	{
		if (ImGui::Button("Auto Connect"))
			app.connect();

		// Manual connect: type the radar IP (e.g. from Retina Finder) and connect directly
		static char host_buf[32] = "";
		ImGui::SetNextItemWidth(140.0f);
		ImGui::InputTextWithHint("##host", "10.1.163.57", host_buf, sizeof(host_buf));
		ImGui::SameLine();
		if (ImGui::Button("Connect to IP") && host_buf[0] != '\0')
			app.connect(host_buf);
	}
	else if (status == MyApp::ConnectionStatus::Finding)
	{
		if (ImGui::Button("Connecting..."))
			app.cancelOrDisconnect();
	}
	else
	{
		if (ImGui::Button("Disconnect"))
			app.cancelOrDisconnect();
	}
}

void SensorWindow::drawSensorInfo()
{
	const auto& frame = MyApp::get().getLastFrame();
	if (frame.deltaUs == 0)
	{
		ImGui::Text("Frame    : N/A");
		ImGui::Text("Bandwidth: N/A");
		ImGui::Text("Update   : N/A");
		ImGui::Text("Packet   : N/A");
		ImGui::Text("Points   : N/A");
		ImGui::Text("Targets  : N/A");
		ImGui::Text("Delta    : N/A");
		return;
	}

	ImGui::Text("Frame    : %u", frame.frameCount);
	ImGui::Text("Bandwidth: %.3f Mbps", MyApp::get().getLastBandwidthMbps());
	ImGui::Text("Update   : %.2f FPS", MyApp::get().getLastFrameRate());
	ImGui::Text("Packet   : %u bytes", frame.packetSize);
	ImGui::Text("Points   : %zu", frame.points.size());
	ImGui::Text("Targets  : %zu", frame.targets.size());
	ImGui::Text("Delta    : %.3f ms", frame.deltaUs / 1000.f);
}

void SensorWindow::drawSensorSpec()
{
	auto& draft = m_draftSensorSpec;
	const auto& applied = MyApp::get().getDeviceInfo().sensorSpec;
	const float label_w = specLabelColumnWidth();
	const float field_w = specFieldWidth(label_w);

	beginSpecRow("fov H:", label_w);
	ImGui::SetNextItemWidth(field_w);
	ImGui::InputFloat("##fovH", &draft.hfovDeg, 0.0f, 0.0f, "%.2f°");
	drawDirtyMark(draft.hfovDeg != applied.hfovDeg);

	beginSpecRow("fov V:", label_w);
	ImGui::SetNextItemWidth(field_w);
	ImGui::InputFloat("##fovV", &draft.vfovDeg, 0.0f, 0.0f, "%.2f°");
	drawDirtyMark(draft.vfovDeg != applied.vfovDeg);

	beginSpecRow("Size:", label_w);
	ImGui::SetNextItemWidth(specVecWidth(field_w, 2));
	ImGui::InputFloat2("##Size", &draft.sensorWidth, "%.2f");
	drawDirtyMark(draft.sensorWidth != applied.sensorWidth || draft.sensorHeight != applied.sensorHeight);

	beginSpecRow("Position:", label_w);
	ImGui::SetNextItemWidth(specVecWidth(field_w, 3));
	ImGui::InputFloat3("##Position", &draft.posX, "%.2f");
	drawDirtyMark(draft.posX != applied.posX || draft.posY != applied.posY || draft.posZ != applied.posZ);

	beginSpecRow("Rotation:", label_w);
	ImGui::SetNextItemWidth(specVecWidth(field_w, 2));
	ImGui::InputFloat2("##Rotation", &draft.yawDeg, "%.1f°");
	drawDirtyMark(draft.yawDeg != applied.yawDeg || draft.pitchDeg != applied.pitchDeg);

	drawMinMaxRow("Range X:", label_w, field_w, "##RangeMinX", "##RangeMaxX", &draft.rangeMinX, &draft.rangeMaxX,
		draft.rangeMinX != applied.rangeMinX || draft.rangeMaxX != applied.rangeMaxX);
	drawMinMaxRow("Range Y:", label_w, field_w, "##RangeMinY", "##RangeMaxY", &draft.rangeMinY, &draft.rangeMaxY,
		draft.rangeMinY != applied.rangeMinY || draft.rangeMaxY != applied.rangeMaxY);
	drawMinMaxRow("Range Z:", label_w, field_w, "##RangeMinZ", "##RangeMaxZ", &draft.rangeMinZ, &draft.rangeMaxZ,
		draft.rangeMinZ != applied.rangeMinZ || draft.rangeMaxZ != applied.rangeMaxZ);

	beginSpecRow("Range:", label_w);
	ImGui::SetNextItemWidth(field_w);
	ImGui::InputFloat("##Range", &draft.range, 0.0f, 0.0f, "%.2f m");
	drawDirtyMark(draft.range != applied.range);

	ImGui::Spacing();
	if (!sensorSpecEqual(draft, applied))
	{
		if (ImGui::Button("Cancel"))
			cancelSensorSpec();
		ImGui::SameLine();
		if (ImGui::Button("Apply"))
			applySensorSpec();
	}
	else
	{
		const bool at_default = sensorSpecEqual(applied, retina::SensorSpec{});
		ImGui::BeginDisabled(at_default);
		if (ImGui::Button("Reset"))
			resetSensorSpec();
		ImGui::EndDisabled();
	}
}

void SensorWindow::applySensorSpec()
{
	MyApp::get().setSensorSpec(m_draftSensorSpec);
}

void SensorWindow::cancelSensorSpec()
{
	m_draftSensorSpec = MyApp::get().getDeviceInfo().sensorSpec;
}

void SensorWindow::resetSensorSpec()
{
	MyApp::get().resetSensorSpec();
	m_draftSensorSpec = MyApp::get().getDeviceInfo().sensorSpec;
}
