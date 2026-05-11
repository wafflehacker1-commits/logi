# LogiInsight Smart Webcam — Python API

Python client library for controlling AI models and reading real-time tracking data (face bounding boxes, landmarks, hand tracking) from the LogiInsight webcam device.

---

## Architecture

The LogiInsight device exposes two interfaces over a single USB-C connection:

| Interface | Protocol | Purpose |
|-----------|----------|---------|
| UVC | USB Video Class (`/dev/videoN`) | Camera frames (up to 1920x1080) |
| USB NCM | Ethernet over USB → SSH (`192.168.1.10:22`) | AI JSON output, device config |

The firmware runs face detection, face mesh (478 landmarks), and hand tracking (21 landmarks) on the device's onboard NPU. Results are written to a JSON file on the device and can be read at any time over SSH.

---

## Files

| File | Description |
|------|-------------|
| `logiwebcam.py` | **Python client library** — SSH connection, AI config, JSON read. |
| `example.py` | Usage examples: live polling and JSON snapshots. |
| `docs/` | API documentation (open `docs/index.html` in a browser). |

---

## Quick Start

### 1. Install

```bash
pip install paramiko opencv-python numpy
```

### 2. Connect and Capture

```python
from logiwebcam import LogiWebcamClient

with LogiWebcamClient(ip="192.168.1.10") as cam:
    cam.set_ai_config(enable_face_detection=True, enable_face_mesh=True)
    data = cam.get_latest_ai_data()
    if data:
        for face in data["faces"]:
            x, y, w, h = face["bbox"]
            print(f"Face at ({x}, {y}) size {w}x{h}, confidence={face['confidence']}")
```

### 3. Run the Example

```bash
python3 example.py                       # 10 frames, face detection
python3 example.py --face-mesh           # + 478-point face landmarks
python3 example.py --hands               # + 21-point hand tracking
python3 example.py --face-mesh --hands   # all AI models enabled
python3 example.py --frames 20           # 20 frames
python3 example.py --width 640 --height 480
```

The example opens the UVC camera, captures MJPG frames, reads the AI JSON from the device, draws face/hand overlays, and saves paired `frame_NNNN.png` + `frame_NNNN.json` files to `./captures/`.

---

## Pre-Requisites

### Hardware

- **LogiInsight webcam** connected via USB-C to the host.
- The device must have booted and the firmware must be running.
- USB NCM network interface configured (device IP: `192.168.1.10`).

### Software

- **Python 3.8+**
- **paramiko**, **opencv-python**, **numpy** (`pip install paramiko opencv-python numpy`)

### Network Setup

```bash
# After plugging in the device, a usb0 interface should appear
ip addr show usb0

# Assign an IP on the same subnet if not auto-configured
sudo ip addr add 192.168.1.20/24 dev usb0
sudo ip link set usb0 up

# Verify connectivity
ping -c 2 192.168.1.10
ssh makentu@192.168.1.10   # password: MakeNTU@123
```

---

## API Reference

### `LogiWebcamClient`

```python
from logiwebcam import LogiWebcamClient

cam = LogiWebcamClient(
    ip="192.168.1.10",         # Device IP via USB NCM
    username="makentu",         # SSH username
    password="MakeNTU@123",     # SSH password
    port=22,                    # SSH port
    timeout=10,                 # Connection timeout (seconds)
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `connect()` | Establish SSH connection to the device. |
| `disconnect()` | Close the SSH connection. |
| `set_ai_config(enable_face_detection, enable_face_mesh, enable_hands)` | Configure which AI models run on the NPU. |
| `get_latest_ai_data()` | Read the latest AI JSON output. Returns `None` if unavailable. |
| `is_connected` (property) | Check if the SSH connection is active. |

#### Context Manager

```python
with LogiWebcamClient() as cam:
    # cam.connect() called automatically
    data = cam.get_latest_ai_data()
# cam.disconnect() called automatically
```

---

## AI Models

| Model | INI Flag | Output | Default |
|-------|----------|--------|---------|
| Face Detection | `enable_face_detection` | Bounding box + confidence | Enabled |
| Face Mesh | `enable_face_mesh` | 478 3D landmarks | Disabled |
| Hand Tracking | `enable_hands` | 21 3D landmarks per hand | Disabled |

> **Note:** Face Mesh requires `enable_face_detection=True` (the mesh model depends on the face detector's bounding box).

---

## AI Output JSON Format

```json
{
  "timestamp_ms": 453404,
  "faces": [
    {
      "bbox": [956.0, 400.0, 264.0, 349.0],
      "confidence": 1.00,
      "landmarks": [
        [110.25, 210.50, 0.12],
        [115.75, 215.30, 0.08],
        ...
      ]
    }
  ],
  "hands": [
    {
      "bbox": [313.0, 385.0, 89.0, 137.0],
      "confidence": 0.87,
      "landmarks": [
        [60.25, 110.50, 0.00],
        [65.75, 115.30, 0.00],
        ...
      ]
    }
  ]
}
```

- **bbox**: `[x, y, width, height]` in sensor pixel coordinates (sensor resolution: 3840x2160)
- **confidence**: detection score [0.0, 1.0]
- **landmarks**: array of `[x, y, z]` arrays in pixel coordinates

### Face Landmarks (478 points)

Key regions: forehead, eyebrows, eyes, irises, nose, lips, jaw contour. See `docs/data_models.html` for the full diagram and index mapping.

### Hand Landmarks (21 points)

| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | WRIST | 11 | MIDDLE_FINGER_DIP |
| 1 | THUMB_CMC | 12 | MIDDLE_FINGER_TIP |
| 2 | THUMB_MCP | 13 | RING_FINGER_MCP |
| 3 | THUMB_IP | 14 | RING_FINGER_PIP |
| 4 | THUMB_TIP | 15 | RING_FINGER_DIP |
| 5 | INDEX_FINGER_MCP | 16 | RING_FINGER_TIP |
| 6 | INDEX_FINGER_PIP | 17 | PINKY_MCP |
| 7 | INDEX_FINGER_DIP | 18 | PINKY_PIP |
| 8 | INDEX_FINGER_TIP | 19 | PINKY_DIP |
| 9 | MIDDLE_FINGER_MCP | 20 | PINKY_TIP |
| 10 | MIDDLE_FINGER_PIP | | |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `SSH connection failed` | Verify `ping 192.168.1.10`. Check that `usb0` has an IP in the `192.168.1.x` range. |
| No data returned | The firmware may not have started writing JSON yet. Wait a few seconds and retry. |
| Device timestamps frozen | Check if the firmware process is running on the device. |
| `landmarks` empty despite `enable_face_mesh=True` | The face mesh model may need a few seconds to load after config change. |

---

## Device Details

- **AI config file (on device):** `/files/logi_conf/logiwebcam_ai.ini`
- **AI output JSON (on device):** `/files/logi_conf/logiwebcam_ai_output.json`
- **SSH credentials:** `makentu` / `MakeNTU@123`
- **Device IP (USB NCM):** `192.168.1.10`
