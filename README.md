# YOLOv8 Pose Tracking for RK3588

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-RK3588-blue.svg)](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)

Real-time pose detection and tracking system optimized for RK3588 platforms.

## Features

- **YOLOv8 Pose Detection** - Real-time human pose estimation with 17 keypoints
- **ByteTrack** - Multi-object tracking with Kalman filtering
- **OSNet ReID** - Person re-identification for stable ID preservation
- **Keypoint Smoothing** - One Euro Filter for smooth output
- **WebSocket Server** - Real-time data streaming
- **RTSP Streaming** - Hardware-accelerated video streaming

## Requirements

### Hardware
- RK3588 platform with NPU support
- USB camera or RTSP stream source

### Software
- OpenCV (≥4.0)
- Eigen3 (≥3.3)
- OpenSSL
- RKNN runtime (included)
- RGA library (included)

## Quick Start

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run
```bash
export LD_LIBRARY_PATH=./librknn_api/aarch64:./mk_api:./mpp_api:$LD_LIBRARY_PATH

# Pose tracking with ByteTrack
./build/pose_track_demo pose_track_config.ini

# RTSP streaming (production)
./build/usb_and_rtsp_to_rtsp_demo yunyan_simple_config_usb.ini
```

### Model Files
Download RKNN model files and place in `weights/`:
- `yolov8s-pose.int.rknn` - YOLOv8 pose model (recommended)
- `osnet_x1_0_imagenet_int8.rknn` - ReID model (optional)

> Convert from PyTorch/ONNX using Rockchip RKNN Toolkit.

## Configuration

Edit `.ini` files to configure:
- `ModelPath` - Path to RKNN model
- `StreamUrl` - Input source (`/dev/video0`, RTSP, or video file)
- `BoxThreshold` - Detection confidence (default: 0.5)
- `EnableTracking` - Enable ByteTrack tracking
- `EnableReID` - Enable person re-identification
- `EnableWebSocket` - Enable WebSocket server
- `WebSocketPort` - WebSocket port (default: 3000)
- `PushRtspPort` - RTSP output port (default: 3554)

## WebSocket Format

Connect to `ws://host:3000`:
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
        ...
      ]
    }
  ]
}
```

## Project Structure

```
YPTR/
├── src/
│   ├── tracking/          # ByteTrack implementation
│   ├── reid/              # OSNet ReID
│   ├── filter/            # Keypoint smoothing
│   ├── task/              # YOLOv8 inference
│   ├── engine/            # RKNN engine
│   ├── process/           # Pre/post-processing
│   ├── websocket/         # WebSocket server
│   └── ...
├── weights/               # Model files (*.rknn)
├── librknn_api/           # RKNN runtime
└── 3rdparty/rga/          # RGA library
```

## Available Demos

| Executable | Description |
|------------|-------------|
| `pose_track_demo` | Pose tracking with ByteTrack + ReID |
| `usb_and_rtsp_to_rtsp_demo` | Multi-threaded RTSP streaming |
| `local_camera_demo` | Local camera with position estimation |
| `img_demo` | Static image detection |

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- [One Euro Filter](https://gery.casiez.net/1euro/)
