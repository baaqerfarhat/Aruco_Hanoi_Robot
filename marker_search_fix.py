# Fix for marker_search: replace markers.update(detections) with this block.
#
# Old code (overwrites earlier detections blindly):
#
#   detections = self.markers_in_base_frame(frame, best_q_deg)
#   markers.update(detections)
#
# New code (keeps the closest observation per marker):
#
#   Also add at top of marker_search:
#       marker_distances: dict[int, float] = {}

cam_pos = (self._kin.fk_base_to_frame(best_q_deg, 5) @ T_5_cam)[:3, 3]
detections = self.markers_in_base_frame(frame, best_q_deg)
for mid, T_base_marker in detections.items():
    dist = float(np.linalg.norm(T_base_marker[:3, 3] - cam_pos[:3]))
    if mid not in markers or dist < marker_distances[mid]:
        markers[mid] = T_base_marker
        marker_distances[mid] = dist
