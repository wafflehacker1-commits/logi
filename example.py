#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A. (LOGITECH) All Rights Reserved. This program is a trade
secret of LOGITECH, and it is not to be reproduced, published, disclosed to others, copied,
adapted, distributed or displayed without the prior authorization of LOGITECH. Licensee agrees
to attach or embed this notice on all copies of the program, including partial copies or
modified versions thereof.

LogiInsight — Example: Open the webcam, capture frames, and read AI data.

This script demonstrates how to use the LogiInsight Python API to:
  1. Open the UVC camera stream (MJPG)
  2. Connect to the device over SSH (USB NCM)
  3. Configure the on-device AI models (face detection, face mesh, hand tracking)
  4. Capture frames and read the AI JSON output simultaneously
  5. Draw face/hand landmarks on the captured frames
  6. Save annotated PNG + JSON pairs

Requirements:
    sudo apt install python3-opencv
    pip install opencv-python paramiko numpy

Usage:
    python3 example.py                       # 10 frames, face detection only
    python3 example.py --frames 20           # 20 frames
    python3 example.py --face-mesh           # enable 478-point face landmarks
    python3 example.py --hands               # enable 21-point hand tracking
    python3 example.py --face-mesh --hands   # enable everything
    python3 example.py --width 640 --height 480
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from logiwebcam import LogiWebcamClient

# ── Face mesh landmark regions (MediaPipe 478-point indices) ─────────────────
FACE_REGIONS = {
    "jaw":       (list(range(172, 200)) + [0],                   (180, 180, 180)),
    "l_eyebrow": (list(range(70, 80)),                           (180, 130, 255)),
    "r_eyebrow": (list(range(300, 310)),                         (180, 130, 255)),
    "l_eye":     (list(range(130, 160)),                          (255, 180, 100)),
    "r_eye":     (list(range(360, 390)),                          (255, 180, 100)),
    "nose":      (list(range(1, 20)),                             (0, 200, 255)),
    "lips":      (list(range(61, 96)) + list(range(185, 213)),    (80, 80, 255)),
    "l_iris":    (list(range(468, 473)),                          (80, 230, 80)),
    "r_iris":    (list(range(473, 478)),                          (80, 230, 80)),
}

# ── Hand skeleton connections (MediaPipe 21-point topology) ──────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

FINGER_COLORS = {
    "thumb": (0, 200, 255), "index": (255, 150, 50), "middle": (50, 220, 50),
    "ring": (50, 50, 255), "pinky": (220, 50, 220), "palm": (200, 200, 200),
}


def _finger_color(i1, i2):
    for idx in (i1, i2):
        if idx in (1, 2, 3, 4):   return FINGER_COLORS["thumb"]
        if idx in (5, 6, 7, 8):   return FINGER_COLORS["index"]
        if idx in (9, 10, 11, 12): return FINGER_COLORS["middle"]
        if idx in (13, 14, 15, 16): return FINGER_COLORS["ring"]
        if idx in (17, 18, 19, 20): return FINGER_COLORS["pinky"]
    return FINGER_COLORS["palm"]


def _lm_xy(lm):
    """Extract (x, y) from a landmark — handles both [x,y,z] and {"x":..} formats."""
    if isinstance(lm, dict):
        return lm["x"], lm["y"]
    return lm[0], lm[1]


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_face_bbox(frame, face, frame_w, frame_h, sensor_w=3840, sensor_h=2160):
    """Draw face bounding box scaled from sensor to frame coordinates."""
    bbox = face.get("bbox", [])
    if len(bbox) < 4:
        return
    bx, by, bw, bh = bbox
    x1, y1 = int(bx), int(by)
    x2, y2 = int(bx + bw), int(by + bh)

    conf = face.get("confidence", 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"face {conf:.2f}", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_face_landmarks(frame, landmarks, frame_w, frame_h):
    """Draw 478 face mesh landmarks, color-coded by region."""
    if not landmarks:
        return
    n = len(landmarks)
    for lm in landmarks:
        lx, ly = _lm_xy(lm)
        x, y = int(lx), int(ly)
        if 0 <= x < frame_w and 0 <= y < frame_h:
            cv2.circle(frame, (x, y), 1, (100, 100, 100), -1)

    for region_name, (indices, color) in FACE_REGIONS.items():
        for idx in indices:
            if idx >= n:
                continue
            lx, ly = _lm_xy(landmarks[idx])
            x, y = int(lx), int(ly)
            if 0 <= x < frame_w and 0 <= y < frame_h:
                r = 3 if "iris" in region_name else 2
                cv2.circle(frame, (x, y), r, color, -1)

    cv2.putText(frame, f"Face Mesh: {n} pts", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_hand_landmarks(frame, hand, frame_w, frame_h):
    """Draw 21-point hand skeleton with colored finger chains."""
    landmarks = hand.get("landmarks", [])
    if not landmarks:
        return
    n = len(landmarks)

    pts = []
    for lm in landmarks:
        lx, ly = _lm_xy(lm)
        pts.append((int(lx), int(ly)))

    for i1, i2 in HAND_CONNECTIONS:
        if i1 >= n or i2 >= n:
            continue
        cv2.line(frame, pts[i1], pts[i2], _finger_color(i1, i2), 2)

    for i, (x, y) in enumerate(pts):
        if 0 <= x < frame_w and 0 <= y < frame_h:
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), r, (0, 0, 0), 1)

    cv2.putText(frame, f"Hand: {n} pts", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ── UVC helpers ──────────────────────────────────────────────────────────────

def find_uvc_device():
    """Auto-detect the first working UVC video device."""
    import platform
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2
    
    for idx in range(10):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"  Found camera at index: {idx}")
                return idx
        cap.release()
    return -1



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogiInsight — Capture UVC frames + AI data"
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Video device path (e.g. /dev/video0). Auto-detect if omitted.")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to capture (default: 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between captures (default: 1.0)")
    parser.add_argument("--output", type=str, default="captures",
                        help="Output directory (default: ./captures)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Frame width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Frame height (default: 1080)")
    parser.add_argument("--face-mesh", action="store_true",
                        help="Enable 478-point face mesh landmarks")
    parser.add_argument("--hands", action="store_true",
                        help="Enable 21-point hand tracking")
    parser.add_argument("--ip", type=str, default="192.168.1.10",
                        help="Device IP (default: 192.168.1.10)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. Connect to device AI via SSH ──────────────────────────────────
    print("[1/3] Connecting to LogiInsight AI ...")
    try:
        cam = LogiWebcamClient(ip=args.ip)
        cam.connect()
        print(f"  Connected to {args.ip}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    cam.set_ai_config(
        enable_face_detection=True,
        enable_face_mesh=args.face_mesh,
        enable_hands=args.hands,
    )
    print(f"  AI config: face_mesh={args.face_mesh}, hands={args.hands}")
    time.sleep(0.5)  # wait for firmware to pick up INI (~200ms)

    # ── 2. Open UVC camera ───────────────────────────────────────────────
    print("[2/3] Opening UVC camera ...")
    dev_index = 0

    if platform.system() == "Darwin":  # macOS
        cap = cv2.VideoCapture(dev_index, cv2.CAP_AVFOUNDATION)
    else:  # Linux
        cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open /dev/video{dev_index}")
        cam.disconnect()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera: /dev/video{dev_index}  {actual_w}x{actual_h} @ {fps:.0f}fps (MJPG)")

    # warm up
    for _ in range(5):
        cap.read()

    # ── 3. Capture loop ──────────────────────────────────────────────────
    print(f"[3/3] Capturing {args.frames} frames ...\n")

    for i in range(args.frames):
        t0 = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [{i+1:3d}/{args.frames}] FAILED to read frame")
            time.sleep(args.interval)
            continue

        # Read AI data from device
        ai_data = cam.get_latest_ai_data()
        t_capture = time.time()

        # Draw overlays
        annotated = frame.copy()
        faces = []
        hands = []

        if ai_data:
            faces = ai_data.get("faces", [])
            hands = ai_data.get("hands", [])

            for face in faces:
                draw_face_bbox(annotated, face, actual_w, actual_h)
                if args.face_mesh:
                    draw_face_landmarks(annotated, face.get("landmarks", []),
                                        actual_w, actual_h)

            if args.hands:
                for hand in hands:
                    draw_hand_landmarks(annotated, hand, actual_w, actual_h)

        # Save PNG + JSON
        base = f"frame_{i:04d}"
        cv2.imwrite(os.path.join(args.output, f"{base}.png"), annotated)

        record = {
            "frame_index": i,
            "host_timestamp": t_capture,
            "host_time_iso": datetime.now().isoformat(),
            "image_file": f"{base}.png",
            "image_resolution": [actual_w, actual_h],
            "device_ai_data": ai_data,
            "capture_latency_ms": round((t_capture - t0) * 1000, 1),
        }
        with open(os.path.join(args.output, f"{base}.json"), "w") as f:
            json.dump(record, f, indent=2)

        # Log
        dev_ts = ai_data.get("timestamp_ms", "N/A") if ai_data else "N/A"
        n_face_lm = len(faces[0].get("landmarks", [])) if faces else 0
        n_hands = len(hands)
        n_hand_lm = len(hands[0].get("landmarks", [])) if hands else 0
        bbox = faces[0].get("bbox", []) if faces else []

        parts = [f"[{i+1:3d}/{args.frames}]  {base}.png  ts={dev_ts}"]
        if bbox:
            parts.append(f"bbox={bbox}")
        if n_face_lm:
            parts.append(f"face_lm={n_face_lm}")
        if n_hands:
            parts.append(f"hands={n_hands}  hand_lm={n_hand_lm}")
        parts.append(f"latency={record['capture_latency_ms']}ms")
        print(f"  {'  '.join(parts)}")

        elapsed = time.time() - t0
        wait = max(0, args.interval - elapsed)
        if wait > 0 and i < args.frames - 1:
            time.sleep(wait)

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    cam.disconnect()

    print(f"\nDone! {args.frames} frames saved to {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
