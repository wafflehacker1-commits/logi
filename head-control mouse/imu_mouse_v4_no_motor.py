#!/usr/bin/env python3

import time
import math
import statistics
import smbus
from evdev import UInput, ecodes as e

MPU_ADDR = 0x68
I2C_BUS = 1

LOOP_DELAY = 0.015
MOVE_SMOOTH_ALPHA = 0.18
GESTURE_SMOOTH_ALPHA = 0.38

ANGLE_DEADZONE_X = 2.5
ANGLE_DEADZONE_Y = 3.0
MAX_STEP_X = 18
MAX_STEP_Y = 14
CURVE = 1.8

GESTURE_WINDOW = 0.35
GESTURE_COOLDOWN = 1.6
AXIS_DOMINANCE = 1.8
AFTER_CLICK_MOVE_PAUSE = 0.4

NOD_RATE_THRESHOLD = 160.0
SHAKE_RATE_THRESHOLD = 180.0

bus = smbus.SMBus(I2C_BUS)
def buzz_pattern(*durations):
    for seconds in durations:
        time.sleep(abs(seconds))


def wake_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)
    time.sleep(0.05)
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x03)
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x00)
    time.sleep(0.05)


def read_word(reg):
    last_err = None
    for _ in range(3):
        try:
            high = bus.read_byte_data(MPU_ADDR, reg)
            low = bus.read_byte_data(MPU_ADDR, reg + 1)
            value = (high << 8) | low
            if value >= 0x8000:
                value -= 65536
            return value
        except OSError as err:
            last_err = err
            time.sleep(0.01)
    raise last_err


def read_motion():
    ax = read_word(0x3B) / 16384.0
    ay = read_word(0x3D) / 16384.0
    az = read_word(0x3F) / 16384.0
    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0
    return ax, ay, az, gx, gy, gz


def accel_to_roll_pitch(ax, ay, az):
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return roll, pitch


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def normalize(v):
    n = norm(v)
    if n < 0.001:
        return [1.0, 0.0]
    return [v[0] / n, v[1] / n]


def subtract_projection(v, axis):
    d = dot(v, axis)
    return [v[0] - d * axis[0], v[1] - d * axis[1]]


def wait_countdown(label, seconds=3):
    print()
    print(label)
    print(f"Hold this pose. Sampling starts in {seconds} seconds.")
    buzz_pattern(0.06)
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    buzz_pattern(0.12)


def robust_average_pose(samples):
    rolls = [s[0] for s in samples]
    pitches = [s[1] for s in samples]
    med_roll = statistics.median(rolls)
    med_pitch = statistics.median(pitches)

    kept = []
    for s in samples:
        dist = math.sqrt(angle_diff(s[0], med_roll) ** 2 + (s[1] - med_pitch) ** 2)
        if dist <= 4.0:
            kept.append(s)

    if len(kept) < max(20, len(samples) // 3):
        kept = samples

    return tuple(sum(s[i] for s in kept) / len(kept) for i in range(8))


def capture_pose(label, sample_count=260):
    wait_countdown(label)
    samples = []

    while len(samples) < sample_count:
        try:
            ax, ay, az, gx, gy, gz = read_motion()
            roll, pitch = accel_to_roll_pitch(ax, ay, az)
            samples.append((roll, pitch, ax, ay, az, gx, gy, gz))
        except OSError as err:
            print(f"I2C calibration read error: {err}")
            time.sleep(0.05)
            continue
        time.sleep(0.01)

    pose = robust_average_pose(samples)
    print(f"Done: roll={pose[0]:.2f}, pitch={pose[1]:.2f}")
    buzz_pattern(0.05, -0.06, 0.05)
    return pose


def pose_delta(pose, neutral_roll, neutral_pitch):
    return [angle_diff(pose[0], neutral_roll), pose[1] - neutral_pitch]


def make_axes(neutral, left, right, up, down):
    neutral_roll, neutral_pitch = neutral[0], neutral[1]

    left_delta = pose_delta(left, neutral_roll, neutral_pitch)
    right_delta = pose_delta(right, neutral_roll, neutral_pitch)
    up_delta = pose_delta(up, neutral_roll, neutral_pitch)
    down_delta = pose_delta(down, neutral_roll, neutral_pitch)

    right_axis = normalize([
        right_delta[0] - left_delta[0],
        right_delta[1] - left_delta[1],
    ])

    up_vec = [
        up_delta[0] - down_delta[0],
        up_delta[1] - down_delta[1],
    ]

    up_axis = normalize(subtract_projection(up_vec, right_axis))

    if norm(up_axis) < 0.001:
        up_axis = [-right_axis[1], right_axis[0]]
        if dot(up_delta, up_axis) < 0:
            up_axis = [-up_axis[0], -up_axis[1]]

    left_limit = max(3.0, abs(dot(left_delta, right_axis)))
    right_limit = max(3.0, abs(dot(right_delta, right_axis)))
    up_limit = max(3.0, abs(dot(up_delta, up_axis)))
    down_limit = max(3.0, abs(dot(down_delta, up_axis)))

    return {
        "neutral_roll": neutral_roll,
        "neutral_pitch": neutral_pitch,
        "right_axis": right_axis,
        "up_axis": up_axis,
        "left_limit": left_limit,
        "right_limit": right_limit,
        "up_limit": up_limit,
        "down_limit": down_limit,
    }


def scaled_axis_to_step(value, negative_limit, positive_limit, deadzone, max_step):
    if abs(value) < deadzone:
        return 0

    limit = positive_limit if value > 0 else negative_limit
    usable = max(0.1, limit - deadzone)
    normalized = min(1.0, (abs(value) - deadzone) / usable)
    step = int(round(max_step * (normalized ** CURVE)))

    if step < 1:
        step = 1

    return step if value > 0 else -step


def virtual_click(ui, button, label):
    print(label)
    ui.write(e.EV_KEY, button, 1)
    ui.syn()
    time.sleep(0.045)
    ui.write(e.EV_KEY, button, 0)
    ui.syn()


class BackAndForthGesture:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_sign = 0
        self.first_time = 0.0
        self.last_trigger_time = 0.0

    def update(self, rate, other_rate, now):
        if now - self.last_trigger_time < GESTURE_COOLDOWN:
            self.first_sign = 0
            return False

        if abs(rate) < self.threshold:
            if self.first_sign != 0 and now - self.first_time > GESTURE_WINDOW:
                self.first_sign = 0
            return False

        if abs(rate) < abs(other_rate) * AXIS_DOMINANCE:
            return False

        sign = 1 if rate > 0 else -1

        if self.first_sign == 0:
            self.first_sign = sign
            self.first_time = now
            return False

        if now - self.first_time > GESTURE_WINDOW:
            self.first_sign = sign
            self.first_time = now
            return False

        if sign == -self.first_sign:
            self.first_sign = 0
            self.last_trigger_time = now
            return True

        return False


def collect_rate_peaks(label, axis_fn, repetitions=3, seconds=1.3):
    print()
    print(label)
    print(f"Do it {repetitions} times. Each sample window is {seconds:.1f}s.")
    peaks = []

    for rep in range(1, repetitions + 1):
        input(f"Press Enter, then perform gesture #{rep}...")
        buzz_pattern(0.05)

        prev_value = None
        prev_time = time.monotonic()
        peak = 0.0
        end_time = prev_time + seconds

        while time.monotonic() < end_time:
            ax, ay, az, gx, gy, gz = read_motion()
            roll, pitch = accel_to_roll_pitch(ax, ay, az)
            value = axis_fn(roll, pitch)
            now = time.monotonic()

            if prev_value is not None:
                dt = max(0.001, now - prev_time)
                peak = max(peak, abs((value - prev_value) / dt))

            prev_value = value
            prev_time = now
            time.sleep(LOOP_DELAY)

        print(f"  peak rate: {peak:.1f} deg/s")
        peaks.append(peak)
        buzz_pattern(0.05, -0.05, 0.05)

    return peaks


def calibrate_gestures(cal):
    nr = cal["neutral_roll"]
    np = cal["neutral_pitch"]
    right_axis = cal["right_axis"]
    up_axis = cal["up_axis"]

    def x_axis(roll, pitch):
        delta = [angle_diff(roll, nr), pitch - np]
        return dot(delta, right_axis)

    def up_axis_value(roll, pitch):
        delta = [angle_diff(roll, nr), pitch - np]
        return dot(delta, up_axis)

    nod_peaks = collect_rate_peaks(
        "Gesture calibration: quick nod up/down for LEFT CLICK.",
        up_axis_value,
    )

    shake_peaks = collect_rate_peaks(
        "Gesture calibration: quick shake left/right for RIGHT CLICK.",
        x_axis,
    )

    nod_peak = statistics.median(nod_peaks)
    shake_peak = statistics.median(shake_peaks)

    nod_threshold = max(150.0, nod_peak * 0.62)
    shake_threshold = max(170.0, shake_peak * 0.62)

    print()
    print(f"Learned nod threshold   = {nod_threshold:.1f} deg/s")
    print(f"Learned shake threshold = {shake_threshold:.1f} deg/s")

    return nod_threshold, shake_threshold


print("Starting MPU6050...")
wake_mpu()

print()
print("Improved calibration:")
print("  1. Neutral")
print("  2. Comfortable LEFT")
print("  3. Comfortable RIGHT")
print("  4. Look UP / cursor-up pose")
print("  5. Look DOWN / cursor-down pose")
print("  6. Intentional nod and shake gesture samples")
print()

neutral = capture_pose("Step 1/5: neutral, look straight forward.")
left = capture_pose("Step 2/5: tilt head comfortably LEFT.")
right = capture_pose("Step 3/5: tilt head comfortably RIGHT.")
up = capture_pose("Step 4/5: look UP, comfortable cursor-up pose.")
down = capture_pose("Step 5/5: look DOWN, comfortable cursor-down pose.")

cal = make_axes(neutral, left, right, up, down)
NOD_RATE_THRESHOLD, SHAKE_RATE_THRESHOLD = calibrate_gestures(cal)

print()
print("Calibration complete.")
print(f"right_axis     = [{cal['right_axis'][0]:.3f}, {cal['right_axis'][1]:.3f}]")
print(f"up_axis        = [{cal['up_axis'][0]:.3f}, {cal['up_axis'][1]:.3f}]")
print(f"left/right lim = {cal['left_limit']:.2f}/{cal['right_limit']:.2f}")
print(f"up/down lim    = {cal['up_limit']:.2f}/{cal['down_limit']:.2f}")
print()

capabilities = {
    e.EV_REL: [e.REL_X, e.REL_Y],
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
}

ui = UInput(capabilities, name="head-imu-mouse-calibrated", version=0x1)

print("Head mouse running:")
print("  Slow head movement -> cursor movement")
print("  Calibrated quick nod -> left click")
print("  Calibrated quick shake -> right click")
print("  Ctrl+C -> stop")
print()

move_x = 0.0
move_up = 0.0
gesture_x = 0.0
gesture_up = 0.0
prev_gesture_x = 0.0
prev_gesture_up = 0.0
prev_time = time.monotonic()

nod_detector = BackAndForthGesture(NOD_RATE_THRESHOLD)
shake_detector = BackAndForthGesture(SHAKE_RATE_THRESHOLD)

pause_move_until = 0.0
i2c_error_count = 0

try:
    while True:
        now = time.monotonic()
        dt = max(LOOP_DELAY, now - prev_time)

        try:
            ax, ay, az, gx, gy, gz = read_motion()
            i2c_error_count = 0
        except OSError as err:
            i2c_error_count += 1
            print(f"I2C read error #{i2c_error_count}: {err}")

            if i2c_error_count >= 5:
                print("Re-waking MPU6050...")
                try:
                    wake_mpu()
                except OSError:
                    pass
                i2c_error_count = 0

            time.sleep(0.05)
            continue

        roll, pitch = accel_to_roll_pitch(ax, ay, az)

        delta = [
            angle_diff(roll, cal["neutral_roll"]),
            pitch - cal["neutral_pitch"],
        ]

        raw_x = dot(delta, cal["right_axis"])
        raw_up = dot(delta, cal["up_axis"])

        move_x = MOVE_SMOOTH_ALPHA * raw_x + (1 - MOVE_SMOOTH_ALPHA) * move_x
        move_up = MOVE_SMOOTH_ALPHA * raw_up + (1 - MOVE_SMOOTH_ALPHA) * move_up

        gesture_x = GESTURE_SMOOTH_ALPHA * raw_x + (1 - GESTURE_SMOOTH_ALPHA) * gesture_x
        gesture_up = GESTURE_SMOOTH_ALPHA * raw_up + (1 - GESTURE_SMOOTH_ALPHA) * gesture_up

        rate_x = (gesture_x - prev_gesture_x) / dt
        rate_up = (gesture_up - prev_gesture_up) / dt

        prev_gesture_x = gesture_x
        prev_gesture_up = gesture_up
        prev_time = now

        gesture_triggered = False

        if nod_detector.update(rate_up, rate_x, now):
            virtual_click(ui, e.BTN_LEFT, "Quick nod -> LEFT CLICK")
            pause_move_until = now + AFTER_CLICK_MOVE_PAUSE
            gesture_triggered = True

        elif shake_detector.update(rate_x, rate_up, now):
            virtual_click(ui, e.BTN_RIGHT, "Quick shake -> RIGHT CLICK")
            pause_move_until = now + AFTER_CLICK_MOVE_PAUSE
            gesture_triggered = True

        high_speed_motion = (
            abs(rate_x) > SHAKE_RATE_THRESHOLD * 0.70
            or abs(rate_up) > NOD_RATE_THRESHOLD * 0.70
        )

        if not gesture_triggered and now >= pause_move_until and not high_speed_motion:
            dx = scaled_axis_to_step(
                move_x,
                cal["left_limit"],
                cal["right_limit"],
                ANGLE_DEADZONE_X,
                MAX_STEP_X,
            )

            dy = -scaled_axis_to_step(
                move_up,
                cal["down_limit"],
                cal["up_limit"],
                ANGLE_DEADZONE_Y,
                MAX_STEP_Y,
            )

            if dx != 0:
                ui.write(e.EV_REL, e.REL_X, dx)

            if dy != 0:
                ui.write(e.EV_REL, e.REL_Y, dy)

            if dx != 0 or dy != 0:
                ui.syn()

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("\nStopping")

finally:
    ui.close()
    print("Virtual mouse closed.")
