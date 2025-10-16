# YOLOv8 Pose Tracking Demo

A real-time pose detection and tracking system for RK3588 platforms, featuring:

- **YOLOv8 Pose Detection** - Real-time human pose estimation with 17 keypoints
- **ByteTrack** - Multi-object tracking with Kalman filtering
- **OSNet ReID** - Person re-identification for robust ID preservation
- **Keypoint Smoothing** - One Euro Filter for stable keypoint output
- **WebSocket Server** - Real-time data streaming to web clients

## Features

### Core Capabilities
- Single-threaded architecture for guaranteed frame order
- Hardware acceleration via RKNN runtime on RK3588 NPU
- Configurable tracking parameters via INI file
- Optional position estimation from pose keypoints
- Real-time WebSocket streaming with JSON format

### Tracking System
The demo implements a robust multi-object tracking pipeline:
- **Motion prediction** using Kalman filter
- **Appearance matching** using OSNet ReID features
- **Occlusion handling** with track buffer
- **Stable ID preservation** across frames

## Requirements

### Hardware
- RK3588 platform with NPU support
- USB camera or RTSP stream source

### Software Dependencies
- OpenCV (≥4.0)
- Eigen3 (≥3.3) - Required for Kalman filter
- OpenSSL - Required for WebSocket server
- RKNN runtime - Included in `librknn_api/`
- RGA library - Included in `3rdparty/rga/`

## Build Instructions

```bash
# Configure CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build

# The executable will be in build/pose_track_demo
```

## Running

```bash
# Set library path
export LD_LIBRARY_PATH=./librknn_api/aarch64:$LD_LIBRARY_PATH

# Run with configuration file
./build/pose_track_demo pose_track_config.ini
```

### Input Sources
- USB Camera: `/dev/video0`, `/dev/video1`, etc.
- RTSP Stream: `rtsp://192.168.1.100:8554/stream`
- Video File: `path/to/video.mp4`

Edit `StreamUrl` in `pose_track_config.ini` to change input source.

## Configuration

Key parameters in `pose_track_config.ini`:

### YOLOv8 Detection
- `ModelPath` - Path to RKNN model (*.rknn)
- `BoxThreshold` - Detection confidence threshold (default: 0.5)
- `NMSThreshold` - Non-maximum suppression threshold (default: 0.65)
- `MinConfidence` - Minimum confidence for tracking (default: 0.25)

### ByteTrack Tracking
- `EnableTracking` - Enable/disable tracking (1/0)
- `TrackBufferFrames` - Frames to keep lost tracks (default: 30)
- `TrackThreshold` - High-confidence detection threshold (default: 0.5)
- `MatchThreshold` - IoU matching threshold (default: 0.8)

### ReID Configuration
- `EnableReID` - Enable/disable ReID (1/0)
- `ReIDModelPath` - Path to OSNet model
- `ReIDInterval` - Feature extraction interval in frames (default: 3)
- `ReIDSimilarityThresh` - Cosine similarity threshold (default: 0.3)

### Keypoint Smoothing
- `EnableSmoothing` - Enable/disable smoothing (1/0)
- `SmoothingMinCutoff` - Smoothing strength (0.5-2.0, default: 1.0)
- `SmoothingBeta` - Motion responsiveness (0.001-0.05, default: 0.007)

### WebSocket Server
- `EnableWebSocket` - Enable/disable WebSocket (1/0)
- `WebSocketHost` - Server host (default: 0.0.0.0)
- `WebSocketPort` - Server port (default: 3000)

## WebSocket Data Format

Connect to `ws://host:port` to receive real-time tracking data:

```json
{
  "balls": [
    {
      "id": 1,
      "x": 0.25,
      "y": 0.0,
      "stable": true,
      "keypoints": [
        {"id": 0, "x": 0.52, "y": 0.31, "score": 0.89},
        {"id": 5, "x": 0.48, "y": 0.45, "score": 0.92},
        ...
      ]
    }
  ]
}
```

- `id` - Stable track ID from ByteTrack
- `x`, `y` - Normalized body center coordinates
- `stable` - Track stability indicator
- `keypoints` - Array of 17 normalized keypoints with confidence scores

## Performance Tuning

### Frequent ID Switches
- Decrease `ReIDInterval` (5→3→2)
- Increase `BoxThreshold` (0.5→0.6)
- Decrease `MinConfidence` (0.3→0.25)

### Low FPS
- Increase `ReIDInterval` (3→5)
- Use int8 quantized models
- Disable `EnableSmoothing`

### Missing Detections
- Decrease `BoxThreshold` (0.5→0.4)
- Decrease `MinConfidence` (0.25→0.2)

## Keyboard Controls (During Runtime)

- `q` / `ESC` - Quit
- `s` - Toggle smoothing on/off
- `+` / `-` - Adjust smoothing strength
- `[` / `]` - Adjust motion responsiveness

## Model Files

Place RKNN model files in `weights/` directory:
- YOLOv8 Pose: `yolov8s-pose.int.rknn` (recommended)
- OSNet ReID: `osnet_x1_0_imagenet_int8.rknn`

## Project Structure

```
pose_track_standalone/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── pose_track_config.ini    # Runtime configuration
├── pose_1_labels_list.txt   # Class labels
├── src/
│   ├── pose_track_demo.cpp  # Main application
│   ├── tracking/            # ByteTrack implementation
│   ├── reid/                # OSNet ReID
│   ├── filter/              # Keypoint smoothing
│   ├── task/                # YOLOv8 inference
│   ├── engine/              # RKNN engine wrapper
│   ├── process/             # Pre/post-processing
│   ├── draw/                # Visualization
│   ├── websocket/           # WebSocket server
│   ├── position/            # Position estimation
│   ├── reconfig/            # Config parser
│   ├── types/               # Data types
│   └── utils/               # Utilities
├── weights/                 # Model files (*.rknn)
├── librknn_api/             # RKNN runtime
└── 3rdparty/rga/            # RGA library

```

## License

This is a standalone extraction from the yolov8-pose-thread-stream project.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLO models
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) - Person ReID
- [One Euro Filter](https://gery.casiez.net/1euro/) - Signal smoothing
