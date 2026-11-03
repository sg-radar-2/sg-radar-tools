#include "implot3d_custom.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <implot3d_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace MyPlot3D
{
	void applyTheme()
	{
		ImPlot3D::StyleColorsDark();
		ImPlot3DStyle& plot_style = ImPlot3D::GetStyle();
		plot_style.PlotPadding = ImVec2(2.0f, 2.0f);
		plot_style.LabelPadding = ImVec2(3.0f, 3.0f);
		plot_style.LegendPadding = ImVec2(6.0f, 6.0f);
		plot_style.LegendInnerPadding = ImVec2(4.0f, 3.0f);
		plot_style.LegendSpacing = ImVec2(8.0f, 4.0f);
		plot_style.ViewScaleFactor = 1.0f;
		plot_style.Colors[ImPlot3DCol_PlotBg] = ImVec4(0.05f, 0.06f, 0.07f, 1.00f);
		plot_style.Colors[ImPlot3DCol_FrameBg] = ImVec4(0.12f, 0.13f, 0.15f, 1.00f);
		plot_style.Colors[ImPlot3DCol_LegendBg] = ImVec4(0.07f, 0.08f, 0.09f, 0.98f);
		plot_style.Colors[ImPlot3DCol_LegendBorder] = ImVec4(0.20f, 0.22f, 0.25f, 0.85f);
	}

	namespace
	{
		constexpr float kViewGizmoMargin = 24.0f;
		constexpr float kViewGizmoArrowLength = 60.0f;
		constexpr float kViewGizmoRingRadius = 65.0f;
		constexpr float kViewGizmoBackgroundRadius = 65.0f;
		constexpr float kViewGizmoDiscRadius = 15.f;
		constexpr float kViewGizmoArrowHoverDistance = 12.0f;
		constexpr float kViewGizmoRingHoverDistance = 10.0f;
		constexpr float kViewGizmoTranslateSensitivity = 0.01f;
		constexpr float kViewGizmoRotateSensitivity = 0.01f;
		constexpr float kViewGizmoZoomSensitivity = 0.08f;

		// Helper function to calculate point depth for proper Z-sorting
		double GetPointDepth(ImPlot3DPoint p) {
			ImPlot3DContext& gp = *ImPlot3D::GetCurrentContext();
			ImPlot3DPlot& plot = *gp.CurrentPlot;

			// Adjust for inverted axes before rotation
			if (ImPlot3D::ImHasFlag(plot.Axes[0].Flags, ImPlot3DAxisFlags_Invert))
				p.x = -p.x;
			if (ImPlot3D::ImHasFlag(plot.Axes[1].Flags, ImPlot3DAxisFlags_Invert))
				p.y = -p.y;
			if (ImPlot3D::ImHasFlag(plot.Axes[2].Flags, ImPlot3DAxisFlags_Invert))
				p.z = -p.z;

			ImPlot3DPoint p_rot = plot.Rotation * p;
			return p_rot.z;
		}

		constexpr int kCircleTexSize = 32;
		constexpr int kVertsPerMarker = 4;
		constexpr int kIndicesPerMarker = 6;
		constexpr int kTrisPerMarker = 2;

		struct CircleSprite
		{
			ImFontAtlasRectId id = ImFontAtlasRectId_Invalid;
			unsigned short packed_x = 0xFFFF;
			unsigned short packed_y = 0xFFFF;
			bool rasterized = false;
		};

		CircleSprite& GetCircleSprite()
		{
			static CircleSprite s_sprite;
			return s_sprite;
		}

		bool RasterizeCircleSprite(ImFontAtlas* atlas, const ImFontAtlasRect& rect)
		{
			ImTextureData* tex = atlas->TexData;
			if (tex == nullptr || tex->Pixels == nullptr)
				return false;

			const float inv = 1.0f / (float)rect.w;
			const float aa = 1.5f * inv * 2.0f;
			const float radius = 1.0f - aa;

			for (int y = 0; y < rect.h; ++y)
			{
				for (int x = 0; x < rect.w; ++x)
				{
					const float fx = ((float)x + 0.5f) * inv * 2.0f - 1.0f;
					const float fy = ((float)y + 0.5f) * inv * 2.0f - 1.0f;
					const float d = ImSqrt(fx * fx + fy * fy);
					float alpha = (radius - d) / aa + 0.5f;
					alpha = ImClamp(alpha, 0.0f, 1.0f);
					const unsigned char a8 = (unsigned char)(alpha * 255.0f + 0.5f);

					void* pixel = tex->GetPixelsAt(rect.x + x, rect.y + y);
					if (tex->Format == ImTextureFormat_RGBA32)
						*(ImU32*)pixel = IM_COL32(255, 255, 255, a8);
					else
						*(unsigned char*)pixel = a8;
				}
			}

			ImFontAtlasTextureBlockQueueUpload(atlas, tex, rect.x, rect.y, rect.w, rect.h);
			return true;
		}

		bool EnsureCircleSpriteUVs(ImVec2* uv0, ImVec2* uv1)
		{
			ImFontAtlas* atlas = ImGui::GetIO().Fonts;
			if (atlas == nullptr)
				return false;

			CircleSprite& sprite = GetCircleSprite();
			ImFontAtlasRect rect;
			if (sprite.id != ImFontAtlasRectId_Invalid && atlas->GetCustomRect(sprite.id, &rect))
			{
				if (!sprite.rasterized || rect.x != sprite.packed_x || rect.y != sprite.packed_y)
				{
					sprite.rasterized = RasterizeCircleSprite(atlas, rect);
					sprite.packed_x = rect.x;
					sprite.packed_y = rect.y;
				}
				*uv0 = rect.uv0;
				*uv1 = rect.uv1;
				return sprite.rasterized;
			}

			sprite.id = atlas->AddCustomRect(kCircleTexSize, kCircleTexSize, &rect);
			if (sprite.id == ImFontAtlasRectId_Invalid)
				return false;

			sprite.rasterized = RasterizeCircleSprite(atlas, rect);
			sprite.packed_x = rect.x;
			sprite.packed_y = rect.y;
			*uv0 = rect.uv0;
			*uv1 = rect.uv1;
			return sprite.rasterized;
		}

		struct ScatterCache
		{
			bool occupied = false;
			const void* data = nullptr;
			int count = 0;
			int stride = 0;
			int offset = 0;
			ImU64 content_key = 0;
			float marker_size = 0.0f;
			ImPlot3DQuat rotation{};
			double range_min[3]{};
			double range_max[3]{};
			ImPlot3DAxisFlags axis_flags[3]{};
			double ndc_scale[3]{};
			float view_scale = 0.0f;
			ImVec2 plot_min{};
			ImVec2 plot_max{};
			ImVec2 uv0{};
			ImVec2 uv1{};
			ImVector<ImDrawVert> vtx;
			ImVector<double> z;
		};

		ScatterCache& GetScatterCache(ImGuiID id)
		{
			ImGuiStorage* storage = ImGui::GetStateStorage();
			void* ptr = storage->GetVoidPtr(id);
			if (ptr == nullptr)
			{
				ScatterCache* state = IM_NEW(ScatterCache)();
				storage->SetVoidPtr(id, state);
				return *state;
			}
			return *static_cast<ScatterCache*>(ptr);
		}

		bool ScatterCacheMatches(
			const ScatterCache& cache,
			const void* data,
			int count,
			int stride,
			int offset,
			ImU64 content_key,
			float marker_size,
			const ImPlot3DPlot& plot,
			const ImVec2& uv0,
			const ImVec2& uv1)
		{
			if (!cache.occupied || content_key == 0)
				return false;
			if (cache.data != data || cache.count != count || cache.stride != stride || cache.offset != offset)
				return false;
			if (cache.content_key != content_key || cache.marker_size != marker_size)
				return false;
			if (cache.rotation != plot.Rotation || cache.view_scale != plot.GetViewScale())
				return false;
			if (cache.plot_min.x != plot.PlotRect.Min.x || cache.plot_min.y != plot.PlotRect.Min.y ||
				cache.plot_max.x != plot.PlotRect.Max.x || cache.plot_max.y != plot.PlotRect.Max.y)
				return false;
			if (cache.uv0.x != uv0.x || cache.uv0.y != uv0.y || cache.uv1.x != uv1.x || cache.uv1.y != uv1.y)
				return false;
			for (int a = 0; a < 3; ++a)
			{
				if (cache.range_min[a] != plot.Axes[a].Range.Min || cache.range_max[a] != plot.Axes[a].Range.Max)
					return false;
				if (cache.axis_flags[a] != plot.Axes[a].Flags || cache.ndc_scale[a] != plot.Axes[a].NDCScale)
					return false;
			}
			return true;
		}

		void StoreScatterCacheKey(
			ScatterCache& cache,
			const void* data,
			int count,
			int stride,
			int offset,
			ImU64 content_key,
			float marker_size,
			const ImPlot3DPlot& plot,
			const ImVec2& uv0,
			const ImVec2& uv1)
		{
			cache.occupied = true;
			cache.data = data;
			cache.count = count;
			cache.stride = stride;
			cache.offset = offset;
			cache.content_key = content_key;
			cache.marker_size = marker_size;
			cache.rotation = plot.Rotation;
			cache.view_scale = plot.GetViewScale();
			cache.plot_min = plot.PlotRect.Min;
			cache.plot_max = plot.PlotRect.Max;
			cache.uv0 = uv0;
			cache.uv1 = uv1;
			for (int a = 0; a < 3; ++a)
			{
				cache.range_min[a] = plot.Axes[a].Range.Min;
				cache.range_max[a] = plot.Axes[a].Range.Max;
				cache.axis_flags[a] = plot.Axes[a].Flags;
				cache.ndc_scale[a] = plot.Axes[a].NDCScale;
			}
		}

		void SubmitCachedMarkers(ImDrawList3D& draw_list_3d, const ScatterCache& cache)
		{
			const int vtx_count = cache.vtx.Size;
			if (vtx_count <= 0)
				return;

			const int marker_count = vtx_count / kVertsPerMarker;
			draw_list_3d.PrimReserve(marker_count * kIndicesPerMarker, vtx_count);

			std::memcpy(draw_list_3d._VtxWritePtr, cache.vtx.Data, (size_t)vtx_count * sizeof(ImDrawVert));
			draw_list_3d._VtxWritePtr += vtx_count;

			const ImDrawIdx base = (ImDrawIdx)draw_list_3d._VtxCurrentIdx;
			for (int i = 0; i < marker_count; ++i)
			{
				const ImDrawIdx b = (ImDrawIdx)(base + i * kVertsPerMarker);
				draw_list_3d._IdxWritePtr[0] = b;
				draw_list_3d._IdxWritePtr[1] = (ImDrawIdx)(b + 1);
				draw_list_3d._IdxWritePtr[2] = (ImDrawIdx)(b + 2);
				draw_list_3d._IdxWritePtr[3] = b;
				draw_list_3d._IdxWritePtr[4] = (ImDrawIdx)(b + 2);
				draw_list_3d._IdxWritePtr[5] = (ImDrawIdx)(b + 3);
				draw_list_3d._IdxWritePtr += 3 * kTrisPerMarker;

				const double depth = cache.z[i];
				draw_list_3d._ZWritePtr[0] = depth;
				draw_list_3d._ZWritePtr[1] = depth;
				draw_list_3d._ZWritePtr += kTrisPerMarker;
			}

			draw_list_3d._VtxCurrentIdx += (unsigned int)vtx_count;
		}

		void RebuildScatterCache(
			ScatterCache& cache,
			void* user,
			const uint8_t* bytes,
			int count,
			int stride,
			int offset,
			ImPlot3DScatterGetter getter,
			float marker_size,
			const ImPlot3DRange& x_range,
			const ImPlot3DRange& y_range,
			const ImPlot3DRange& z_range,
			const ImVec2& uv0,
			const ImVec2& uv1)
		{
			cache.vtx.resize(0);
			cache.z.resize(0);
			cache.vtx.reserve(count * kVertsPerMarker);
			cache.z.reserve(count);

			const ImVec2 uv01(uv1.x, uv0.y);
			const ImVec2 uv10(uv0.x, uv1.y);

			for (int i = 0; i < count; ++i)
			{
				ImPlot3DPoint p_plot;
				ImU32 col_u32;
				getter(user, bytes + (size_t)(offset + i) * stride, &p_plot, &col_u32);

				if ((col_u32 & IM_COL32_A_MASK) == 0)
					continue;
				if (!x_range.Contains(p_plot.x) || !y_range.Contains(p_plot.y) || !z_range.Contains(p_plot.z))
					continue;

				const ImVec2 p_screen = ImPlot3D::PlotToPixels(p_plot);
				const double point_depth = GetPointDepth(p_plot);

				ImDrawVert v0, v1, v2, v3;
				v0.pos = ImVec2(p_screen.x - marker_size, p_screen.y - marker_size);
				v1.pos = ImVec2(p_screen.x + marker_size, p_screen.y - marker_size);
				v2.pos = ImVec2(p_screen.x + marker_size, p_screen.y + marker_size);
				v3.pos = ImVec2(p_screen.x - marker_size, p_screen.y + marker_size);
				v0.uv = uv0;
				v1.uv = uv01;
				v2.uv = uv1;
				v3.uv = uv10;
				v0.col = v1.col = v2.col = v3.col = col_u32;

				cache.vtx.push_back(v0);
				cache.vtx.push_back(v1);
				cache.vtx.push_back(v2);
				cache.vtx.push_back(v3);
				cache.z.push_back(point_depth);
			}
		}

		enum GizmoPart
		{
			GizmoPart_None = -1,
			GizmoPart_XArrow = 0,
			GizmoPart_YArrow = 1,
			GizmoPart_ZArrow = 2,
			GizmoPart_XRing = 3,
			GizmoPart_YRing = 4,
			GizmoPart_ZRing = 5,
			GizmoPart_Center = 6,
			GizmoPart_Orbit = 7,
		};

		struct GizmoState
		{
			int ActivePart = GizmoPart_None;
			int HoveredPart = GizmoPart_None;
			ImVec2 LastMousePos = ImVec2(0.0f, 0.0f);
			bool HasLastMousePos = false;
			bool CenterPressed = false;
			bool HasInitialized = false;
			bool HasCustomInitialRotation = false;
			bool HasCustomInitialAxesBox = false;
			bool HasCustomInitialZoom = false;

			ImPlot3DQuat InitialRotation = ImPlot3DQuat(-0.513269, -0.212596, -0.318184, 0.76819);
			ImPlot3DBox InitialAxesBox = ImPlot3DBox(ImPlot3DPoint(-10.0, -10.0, -10.0), ImPlot3DPoint(10.0, 10.0, 10.0));
			double InitialZoom = 1.0;

			ImPlot3DQuat Rotation = ImPlot3DQuat(-0.513269, -0.212596, -0.318184, 0.76819);
			bool HasPendingRotation = false;
			bool NeedAnimateRotation = false;

			ImPlot3DBox AxesBox = ImPlot3DBox(ImPlot3DPoint(-10.0, -10.0, -10.0), ImPlot3DPoint(10.0, 10.0, 10.0));
			bool HasPendingAxisLimit = false;

			double Zoom = 1.0;
			bool HasPendingZoom = false;

			ImPlot3DPoint DragRotationAxis = ImPlot3DPoint(0.0, 0.0, 0.0);
		};

		GizmoState& GetGizmoState(ImGuiID id)
		{
			ImGuiStorage* storage = ImGui::GetStateStorage();
			void* ptr = storage->GetVoidPtr(id);
			if (ptr == nullptr)
			{
				GizmoState* state = IM_NEW(GizmoState)();
				storage->SetVoidPtr(id, state);
				return *state;
			}
			return *static_cast<GizmoState*>(ptr);
		}

		ImVec4 AxisColor(int axis)
		{
			switch (axis)
			{
				case 0: return ImVec4(0.93f, 0.26f, 0.30f, 1.0f);
				case 1: return ImVec4(0.40f, 0.82f, 0.19f, 1.0f);
				default: return ImVec4(0.23f, 0.60f, 0.96f, 1.0f);
			}
		}

		ImVec2 toScreen(const ImPlot3DPlot& plot, const ImVec2& center, const ImPlot3DPoint& point)
		{
			ImPlot3DPoint point_pix = plot.Rotation * point;
			point_pix.y *= -1.0f; // Invert y-axis
			point_pix.x += center.x;
			point_pix.y += center.y;

			return ImVec2((float)point_pix.x, (float)point_pix.y);
		}

		ImVec4 AxisColorHovered(const ImVec4& color)
		{
			return ImVec4(
				std::min(color.x + 0.18f, 1.0f),
				std::min(color.y + 0.18f, 1.0f),
				std::min(color.z + 0.18f, 1.0f),
				1.0f);
		}

		float DistanceToSegment(const ImVec2& point, const ImVec2& a, const ImVec2& b)
		{
			const ImVec2 ab(b.x - a.x, b.y - a.y);
			const float length_sq = ab.x * ab.x + ab.y * ab.y;
			
			if (length_sq <= 1.0e-6f)
			{
				const ImVec2 diff(point.x - a.x, point.y - a.y);
				return ImSqrt(diff.x * diff.x + diff.y * diff.y);
			}
			
			const ImVec2 ap(point.x - a.x, point.y - a.y);
			const float t = std::clamp((ap.x * ab.x + ap.y * ab.y) / length_sq, 0.0f, 1.0f);
			const ImVec2 closest(a.x + ab.x * t, a.y + ab.y * t);
			const ImVec2 diff(point.x - closest.x, point.y - closest.y);
			
			return ImSqrt(diff.x * diff.x + diff.y * diff.y);
		}

		ImPlot3DPoint AxisUnit(int axis)
		{
			switch (axis)
			{
				case 0: return ImPlot3DPoint(1.0, 0.0, 0.0);
				case 1: return ImPlot3DPoint(0.0, 1.0, 0.0);
				default: return ImPlot3DPoint(0.0, 0.0, 1.0);
			}
		}

		ImVec2 DrawAxisArrow(const ImPlot3DPlot& plot, ImDrawList* draw_list, const ImVec2& center, const ImPlot3DPoint& dir, const ImVec4& color, float thickness)
		{
			const ImVec2 end = toScreen(plot, center, dir * kViewGizmoArrowLength);
			const ImU32 col = ImGui::ColorConvertFloat4ToU32(color);
			draw_list->AddLine(center, end, col, thickness);
			const ImVec2 delta = ImVec2(end.x - center.x, end.y - center.y);
			const float len = ImLengthSqr(delta) > 0.0f ? ImSqrt(ImLengthSqr(delta)) : 1.0f;
			const ImVec2 tangent(delta.x / len, delta.y / len);
			const ImVec2 normal(-tangent.y, tangent.x);
			const ImVec2 tip0(end.x - tangent.x * 7.0f + normal.x * 3.5f, end.y - tangent.y * 7.0f + normal.y * 3.5f);
			const ImVec2 tip1(end.x - tangent.x * 7.0f - normal.x * 3.5f, end.y - tangent.y * 7.0f - normal.y * 3.5f);
			draw_list->AddTriangleFilled(end, tip0, tip1, col);
			return end;
		}

		void BuildAxisRingPoints(const ImPlot3DPlot& plot, const ImVec2& center, const ImAxis3D& axis, ImVec2* points)
		{
			for (int i = 0; i < 49; ++i)
			{
				const float t = (float)i / 48.0f * 2.0f * IM_PI;
				ImPlot3DPoint local;

				if (axis == ImAxis3D_X)
					local = ImPlot3DPoint(0.0, cos(t), sin(t));
				else if (axis == ImAxis3D_Y)
					local = ImPlot3DPoint(cos(t), 0.0, sin(t));
				else if (axis == ImAxis3D_Z)
					local = ImPlot3DPoint(cos(t), sin(t), 0.0);

				points[i] = toScreen(plot, center, local * kViewGizmoRingRadius);
			}
		}

		void DrawAxisRing(const ImPlot3DPlot& plot, ImDrawList* draw_list, const ImVec2& center, const ImAxis3D& axis, const ImVec4& color, float thickness)
		{
			ImVec2 points[49];
			BuildAxisRingPoints(plot, center, axis, points);
			draw_list->AddPolyline(points, 49, ImGui::ColorConvertFloat4ToU32(color), ImDrawFlags_None, thickness);
		}

		float DistanceToRing(const ImPlot3DPlot& plot, const ImVec2& center, const ImAxis3D& axis, const ImVec2& mouse_pos)
		{
			ImVec2 points[49];
			BuildAxisRingPoints(plot, center, axis, points);
			float best = FLT_MAX;
			for (int i = 1; i < 49; ++i)
				best = std::min(best, DistanceToSegment(mouse_pos, points[i - 1], points[i]));
			return best;
		}

		float SignedAxisDelta(const ImVec2& mouse_delta, const ImVec2& axis_screen)
		{
			const float axis_length_sq = axis_screen.x * axis_screen.x + axis_screen.y * axis_screen.y;
			if (axis_length_sq <= 1.0e-6f)
				return 0.0f;
			return (mouse_delta.x * axis_screen.x + mouse_delta.y * axis_screen.y) / std::sqrt(axis_length_sq);
		}

		ImPlot3DQuat AxisDragRotation(const ImPlot3DQuat& rotation, int axis, float delta)
		{
			return ImPlot3DQuat((double)delta * kViewGizmoRotateSensitivity, rotation * AxisUnit(axis));
		}

		float SignedRingDelta(const ImVec2& center, const ImVec2& mouse_pos, const ImVec2& mouse_delta)
		{
			const ImVec2 radial(mouse_pos.x - center.x, mouse_pos.y - center.y);
			const float radial_len_sq = radial.x * radial.x + radial.y * radial.y;

			if (radial_len_sq <= 1.0e-6f)
				return 0.0f;
			
			const ImVec2 tangent(-radial.y, radial.x);
			const float tangent_len = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
			
			if (tangent_len <= 1.0e-6f)
				return 0.0f;
			
			return -(mouse_delta.x * tangent.x + mouse_delta.y * tangent.y) / tangent_len;
		}

		GizmoPart getHoveredGizmoPart(const ImPlot3DPlot& plot, const ImVec2& center, const ImVec2& mouse_pos, bool hovered)
		{
			const ImVec2 center_delta(mouse_pos.x - center.x, mouse_pos.y - center.y);
			const float center_distance = std::sqrt(center_delta.x * center_delta.x + center_delta.y * center_delta.y);
			
			GizmoPart part = GizmoPart_None;
			float best_distance = FLT_MAX;

			if (center_distance <= kViewGizmoDiscRadius)
			{
				best_distance = 0.0f;
				part = GizmoPart_Center;
			}

			for (int axis = 0; axis < 3; ++axis)
			{
				const ImPlot3DPoint axis_dir = AxisUnit(axis);
				const ImVec2 axis_end = toScreen(plot, center, axis_dir * kViewGizmoArrowLength);
				const float arrow_distance = DistanceToSegment(mouse_pos, center, axis_end);
				const float ring_distance = DistanceToRing(plot, center, (ImAxis3D)axis, mouse_pos);

				if (arrow_distance < kViewGizmoArrowHoverDistance && arrow_distance < best_distance)
				{
					best_distance = arrow_distance;
					part = static_cast<GizmoPart>(GizmoPart_XArrow + axis);
				}

				if (ring_distance < kViewGizmoRingHoverDistance && ring_distance < best_distance)
				{
					best_distance = ring_distance;
					part = static_cast<GizmoPart>(GizmoPart_XRing + axis);
				}
			}

			if (hovered && part == GizmoPart_None)
			{
				part = GizmoPart_Orbit;
			}

			return part;
		}
	}

	// Billboarded circular markers: one atlas sprite quad per point (4 vtx / 2 tri).
	void PlotScatter(
		const char* label,
		void* user,
		const void* data,
		int count,
		int stride,
		ImPlot3DScatterGetter getter,
		const ImPlot3DSpec& spec,
		ImU64 content_key)
	{
		if (count < 1)
			return;

		IM_ASSERT(data != nullptr && "PlotScatter(custom) requires valid data!");
		IM_ASSERT(getter != nullptr && "PlotScatter(custom) requires a valid getter callback!");
		IM_ASSERT(stride > 0 && "PlotScatter(custom) requires stride > 0!");

		ImPlot3DContext& gp = *ImPlot3D::GetCurrentContext();
		IM_ASSERT_USER_ERROR(gp.CurrentPlot != nullptr, "PlotScatter(custom) needs to be called between BeginPlot() and EndPlot()!");

		ImPlot3DPlot& plot = *gp.CurrentPlot;
		ImDrawList3D& draw_list_3d = plot.DrawList;

		const float marker_size = spec.MarkerSize >= 0.0f ? spec.MarkerSize : gp.Style.MarkerSize;

		ImVec2 uv0 = draw_list_3d._SharedData->TexUvWhitePixel;
		ImVec2 uv1 = uv0;
		EnsureCircleSpriteUVs(&uv0, &uv1);

		ScatterCache& cache = GetScatterCache(ImGui::GetID(label));
		if (ScatterCacheMatches(cache, data, count, stride, spec.Offset, content_key, marker_size, plot, uv0, uv1))
		{
			SubmitCachedMarkers(draw_list_3d, cache);
			return;
		}

		RebuildScatterCache(
			cache,
			user,
			static_cast<const uint8_t*>(data),
			count,
			stride,
			spec.Offset,
			getter,
			marker_size,
			plot.Axes[0].Range,
			plot.Axes[1].Range,
			plot.Axes[2].Range,
			uv0,
			uv1);

		StoreScatterCacheKey(cache, data, count, stride, spec.Offset, content_key, marker_size, plot, uv0, uv1);
		SubmitCachedMarkers(draw_list_3d, cache);
	}

	void PlotBox(
		const char* label,
		double minx,
		double maxx,
		double miny,
		double maxy,
		double minz,
		double maxz,
		const ImPlot3DSpec& spec)
	{
		double xs[24] = {
			minx, maxx, maxx, minx,
			minx, minx, maxx, maxx,
			minx, minx, minx, minx,
			maxx, maxx, maxx, maxx,
			minx, maxx, maxx, minx,
			minx, minx, maxx, maxx 
		};
		double ys[24] = {
			miny, miny, miny, miny,
			maxy, maxy, maxy, maxy,
			miny, miny, maxy, maxy,
			miny, maxy, maxy, miny,
			miny, miny, maxy, maxy,
			miny, maxy, maxy, miny 
		};
		double zs[24] = {
			minz, minz, maxz, maxz,
			minz, maxz, maxz, minz,
			minz, maxz, maxz, minz,
			minz, minz, maxz, maxz,
			minz, minz, minz, minz,
			maxz, maxz, maxz, maxz 
		};

		ImPlot3D::PlotQuad(label, xs, ys, zs, 24, spec);
	}

	ImPlot3DViewGizmoFlags ShowViewGizmo(
		const char* label,
		float size,
		ImPlot3DQuat* rotation,
		ImPlot3DBox* axes_box,
		double* zoom)
	{
		ImPlot3DContext& gp = *ImPlot3D::GetCurrentContext();

		IM_ASSERT_USER_ERROR(gp.CurrentPlot != nullptr, "ShowViewGizmo() needs to be called between BeginPlot() and EndPlot()!");
		
		const ImGuiID id = ImGui::GetID(label);
		ImPlot3DPlot& plot = *gp.CurrentPlot;
		GizmoState& state = GetGizmoState(id);

		ImGui::KeepAliveID(id);

		if (!state.HasPendingRotation && rotation && state.Rotation != *rotation)
		{
			state.Rotation = *rotation;
			state.HasPendingRotation = true;
			state.NeedAnimateRotation = true;
		}
		if (!state.HasPendingAxisLimit && axes_box && (state.AxesBox.Min != axes_box->Min || state.AxesBox.Max != axes_box->Max))
		{
			state.AxesBox = *axes_box;
			state.HasPendingAxisLimit = true;
		}
		if (!state.HasPendingZoom && zoom && state.Zoom != *zoom)
		{
			state.Zoom = *zoom;
			state.HasPendingZoom = true;
		}

		if (!state.HasInitialized)
		{
			if (rotation)
				state.Rotation = *rotation;
			else
				state.Rotation = plot.Rotation;
			if (!state.HasCustomInitialRotation)
				state.InitialRotation = state.Rotation;

			if (axes_box)
				state.AxesBox = *axes_box;
			else
			{
				state.AxesBox.Min.x = plot.Axes[0].Range.Min;
				state.AxesBox.Max.x = plot.Axes[0].Range.Max;
				state.AxesBox.Min.y = plot.Axes[1].Range.Min;
				state.AxesBox.Max.y = plot.Axes[1].Range.Max;
				state.AxesBox.Min.z = plot.Axes[2].Range.Min;
				state.AxesBox.Max.z = plot.Axes[2].Range.Max;
			}
			if (!state.HasCustomInitialAxesBox)
				state.InitialAxesBox = state.AxesBox;

			if (zoom)
				state.Zoom = *zoom;
			if (!state.HasCustomInitialZoom)
				state.InitialZoom = state.Zoom;

			state.HasPendingRotation = true;
			state.HasPendingAxisLimit = true;
			state.HasPendingZoom = true;
			state.HasInitialized = true;
		}

		ImPlot3D::SetupBoxInitialRotation(state.InitialRotation);

		// Apply pending setup changes before GetPlotRectPos()/GetPlotRectSize(), which lock plot setup.
		// Changes made by input below are applied on the next frame.
		if (state.HasPendingRotation)
		{
			ImPlot3D::SetupBoxRotation(state.Rotation, state.NeedAnimateRotation, ImPlot3DCond_Always);
			state.NeedAnimateRotation = false;

			if (rotation)
				*rotation = state.Rotation;

			state.HasPendingRotation = false;
		}
		if (state.HasPendingAxisLimit)
		{
			ImPlot3D::SetupAxesLimits(
				state.AxesBox.Min.x,
				state.AxesBox.Max.x,
				state.AxesBox.Min.y,
				state.AxesBox.Max.y,
				state.AxesBox.Min.z,
				state.AxesBox.Max.z,
				ImPlot3DCond_Always);

			if (axes_box)
				*axes_box = state.AxesBox;

			state.HasPendingAxisLimit = false;
		}
		if (state.HasPendingZoom)
		{
			const double x_size = state.AxesBox.Max.x - state.AxesBox.Min.x;
			const double y_size = state.AxesBox.Max.y - state.AxesBox.Min.y;
			const double z_size = state.AxesBox.Max.z - state.AxesBox.Min.z;
			const double xy_size = x_size > 0.0 ? y_size / x_size : 1.0;
			const double xz_size = x_size > 0.0 ? z_size / x_size : 1.0;

			ImPlot3D::SetupBoxScale(state.Zoom, xy_size * state.Zoom, xz_size * state.Zoom);

			if (zoom)
				*zoom = state.Zoom;

			state.HasPendingZoom = false;
		}

		ImDrawList* draw_list = ImGui::GetWindowDrawList();
		const ImGuiIO& io = ImGui::GetIO();

		const ImVec2 plot_pos = ImPlot3D::GetPlotRectPos();
		const ImVec2 plot_size = ImPlot3D::GetPlotRectSize();
		const ImVec2 gizmo_size(size, size);
		const ImVec2 gizmo_min(plot_pos.x + plot_size.x - gizmo_size.x - kViewGizmoMargin, plot_pos.y + kViewGizmoMargin);
		const ImVec2 gizmo_max(gizmo_min.x + gizmo_size.x, gizmo_min.y + gizmo_size.y);
		const ImRect rect(gizmo_min, gizmo_max);
		const ImVec2 center((gizmo_min.x + gizmo_max.x) * 0.5f, (gizmo_min.y + gizmo_max.y) * 0.5f);

		const bool hovered = rect.Contains(io.MousePos);
		const bool was_active = ImGui::GetActiveID() == id;
		const bool active = was_active || (hovered && (ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseDown(ImGuiMouseButton_Right)));

		ImPlot3DViewGizmoFlags result_flags = ImPlot3DViewGizmoFlags_None;

		state.HoveredPart = getHoveredGizmoPart(plot, center, io.MousePos, hovered);

		const int visible_hovered_part = state.ActivePart != GizmoPart_None ? state.ActivePart : state.HoveredPart;

		if (hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)))
		{
			ImGui::SetActiveID(id, ImGui::GetCurrentWindow());
			state.ActivePart = ImGui::IsMouseClicked(ImGuiMouseButton_Right) ? GizmoPart_Orbit : state.HoveredPart;
			state.LastMousePos = io.MousePos;
			state.HasLastMousePos = true;
			state.CenterPressed = state.ActivePart == GizmoPart_Center;
			state.DragRotationAxis = ImPlot3DPoint(0.0, 0.0, 0.0);
		}
		if (was_active && !ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Right))
		{
			if (state.ActivePart == GizmoPart_Center && state.CenterPressed && state.HoveredPart == GizmoPart_Center)
			{
				state.Rotation = state.InitialRotation;
				state.AxesBox = state.InitialAxesBox;
				state.Zoom = state.InitialZoom;
				state.HasPendingAxisLimit = true;
				state.HasPendingRotation = true;
				state.HasPendingZoom = true;
				result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_Reset);
			}

			ImGui::ClearActiveID();
			state.ActivePart = GizmoPart_None;
			state.HasLastMousePos = false;
			state.CenterPressed = false;
			state.DragRotationAxis = ImPlot3DPoint(0.0, 0.0, 0.0);
		}

		if (state.ActivePart != GizmoPart_None && state.ActivePart != GizmoPart_Center && state.HasLastMousePos)
		{
			const ImVec2 delta(io.MousePos.x - state.LastMousePos.x, io.MousePos.y - state.LastMousePos.y);
			state.LastMousePos = io.MousePos;

			if (ImLengthSqr(delta) > 0.0f)
			{
				if (state.ActivePart >= GizmoPart_XArrow && state.ActivePart <= GizmoPart_ZArrow)
				{
					const int axis = state.ActivePart - GizmoPart_XArrow;
					const ImVec2 axis_end = toScreen(plot, center, AxisUnit(axis) * kViewGizmoArrowLength);
					const ImVec2 axis_screen(axis_end.x - center.x, axis_end.y - center.y);
					const float signed_delta = -SignedAxisDelta(delta, axis_screen);
					state.AxesBox.Min[axis] += signed_delta * kViewGizmoTranslateSensitivity;
					state.AxesBox.Max[axis] += signed_delta * kViewGizmoTranslateSensitivity;
					state.HasPendingAxisLimit = true;
					result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_BoxChanged);
				}
				else if (state.ActivePart >= GizmoPart_XRing && state.ActivePart <= GizmoPart_ZRing)
				{
					const int axis = state.ActivePart - GizmoPart_XRing;
					const float signed_delta = SignedRingDelta(center, io.MousePos, delta);
					const ImPlot3DQuat axis_rotation = AxisDragRotation(state.Rotation, axis, signed_delta);
					state.Rotation = axis_rotation * state.Rotation;
					state.Rotation.Normalize();
					state.HasPendingRotation = true;
					result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_RotationChanged);
				}
				else if (state.ActivePart == GizmoPart_Orbit)
				{
					if (state.DragRotationAxis.x == 0.0 && state.DragRotationAxis.y == 0.0 && state.DragRotationAxis.z == 0.0)
					{
						const ImPlot3DPoint up_vector = state.Rotation * ImPlot3DPoint(0.0, 0.0, 1.0);
						state.DragRotationAxis = up_vector.z < 0.0 ? ImPlot3DPoint(0.0, 0.0, -1.0) : ImPlot3DPoint(0.0, 0.0, 1.0);
					}
					const ImPlot3DQuat drag_x((double)delta.y * kViewGizmoRotateSensitivity, ImPlot3DPoint(1.0, 0.0, 0.0));
					const ImPlot3DQuat drag_z((double)delta.x * kViewGizmoRotateSensitivity, state.DragRotationAxis);
					state.Rotation = drag_x * state.Rotation * drag_z;
					state.Rotation.Normalize();
					state.HasPendingRotation = true;
					result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_RotationChanged);
				}
			}
		}

		if ((hovered || was_active) && io.MouseWheel != 0.0f)
		{
			const int wheel_part = state.ActivePart != GizmoPart_None ? state.ActivePart : state.HoveredPart;

			if (wheel_part >= GizmoPart_XRing && wheel_part <= GizmoPart_ZRing)
			{
				const int axis = wheel_part - GizmoPart_XRing;
				const ImPlot3DQuat axis_rotation = AxisDragRotation(state.Rotation, axis, io.MouseWheel * 16.0f);
				state.Rotation = axis_rotation * state.Rotation;
				state.Rotation.Normalize();
				state.HasPendingRotation = true;
				result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_RotationChanged);
			}
			else if (wheel_part == GizmoPart_Center)
			{
				const float mag = 1.0f + io.MouseWheel * kViewGizmoZoomSensitivity;
				state.Zoom = std::clamp(state.Zoom * std::max(0.2f, mag), 0.2, 8.0);
				state.HasPendingZoom = true;
				result_flags = (ImPlot3DViewGizmoFlags)(result_flags | ImPlot3DViewGizmoFlags_ScaleChanged);
			}
		}

		const bool center_hovered = visible_hovered_part == GizmoPart_Center;
		const bool center_active = state.ActivePart == GizmoPart_Center;
		if ((hovered || active) && !center_hovered && !center_active)
		{
			draw_list->AddCircleFilled(center, kViewGizmoBackgroundRadius, ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.2f)), 64);
			draw_list->AddCircle(center, kViewGizmoBackgroundRadius, ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.30f)), 64, 1.5f);
		}

		for (int axis = 0; axis < 3; ++axis)
		{
			const bool ring_hovered = visible_hovered_part == GizmoPart_XRing + axis;
			const bool ring_active = state.ActivePart == GizmoPart_XRing + axis;
			const bool arrow_hovered = visible_hovered_part == GizmoPart_XArrow + axis;
			const bool arrow_active = state.ActivePart == GizmoPart_XArrow + axis;
			const ImVec4 base = AxisColor(axis);
			const ImVec4 ring_color = (ring_hovered || ring_active) ? AxisColorHovered(base) : base;
			const ImVec4 arrow_color = (arrow_hovered || arrow_active) ? AxisColorHovered(base) : base;
			DrawAxisRing(plot, draw_list, center, (ImAxis3D)axis, ring_color, ring_active ? 4.0f : ring_hovered ? 3.2f : 2.0f);
			DrawAxisArrow(plot, draw_list, center, AxisUnit(axis), arrow_color, arrow_active ? 4.0f : arrow_hovered ? 3.2f : 2.0f);
		}

		if (center_hovered || center_active)
		{
			draw_list->AddCircleFilled(center, kViewGizmoDiscRadius, IM_COL32(255, 255, 255, 64), 0);
			draw_list->AddCircle(center, kViewGizmoDiscRadius, IM_COL32(255, 255, 255, 110), 0, 1.5f);
		}

		draw_list->AddCircle(center, kViewGizmoDiscRadius, IM_COL32(255, 255, 255, center_hovered || center_active ? 255 : 220), 0, center_active ? 3.0f : center_hovered ? 2.5f : 2.0f);

		return result_flags;
	}

	void SetViewGizmoInitialRotation(const char* label, const ImPlot3DQuat& rotation)
	{
		const ImGuiID id = ImGui::GetID(label);
		GizmoState& state = GetGizmoState(id);

		state.InitialRotation = rotation;
		state.HasCustomInitialRotation = true;
	}

	void SetViewGizmoInitialAxesBox(const char* label, const ImPlot3DBox& axes_box)
	{
		const ImGuiID id = ImGui::GetID(label);
		GizmoState& state = GetGizmoState(id);

		state.InitialAxesBox = axes_box;
		state.HasCustomInitialAxesBox = true;
	}

	void SetViewGizmoInitialZoom(const char* label, double zoom)
	{
		const ImGuiID id = ImGui::GetID(label);
		GizmoState& state = GetGizmoState(id);

		state.InitialZoom = zoom;
		state.HasCustomInitialZoom = true;
	}
} // namespace MyPlot3D
