# Published continuously, also when empty, so consumers can detect liveness.
std_msgs/Header header             # stamp: image capture time, frame_id: map
string clip_model
ObservedObject[] objects
