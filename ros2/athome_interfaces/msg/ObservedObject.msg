int64 object_id                    # persistent map object ID, -1 for dynamic objects
bool is_static
string label
float32 confidence
geometry_msgs/Point centroid       # map frame
geometry_msgs/Pose bbox_center     # map frame, only yaw is used
geometry_msgs/Vector3 bbox_size    # length, width, height
float32[] clip_feature             # L2-normalized
