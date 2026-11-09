import pygame
import math
import serial
import time
import cv2
import numpy as np
import onnxruntime as ort

from main import RPR


MODEL_PATH = "best_model2.onnx"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.6
CENTER_ZONE_FRACTION = 0.40
# A model detection must also contain a meaningful red area. This suppresses
# false positives from uncoloured tables/backgrounds without drawing anything.
RED_MIN_FRACTION = 0.02
RED_MIN_PIXELS = 150


def apple_in_center(session, input_name, frame):
    """Detect a red apple whose centre is inside the centre zone."""
    rgb_image = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)

    # Red wraps around both ends of the OpenCV HSV hue scale, so use two masks.
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv_image, (0, 80, 50), (10, 255, 255)),
        cv2.inRange(hsv_image, (170, 80, 50), (180, 255, 255)),
    )

    image = rgb_image.astype(np.float32) / 255.0
    image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
    predictions = np.squeeze(session.run(None, {input_name: image})[0])
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes, scores = [], []
    for detection in predictions:
        if len(detection) < 5:
            continue
        x, y, width, height = detection[:4]
        score = float(np.max(detection[4:]))
        if score >= CONF_THRESHOLD:
            boxes.append([int(x - width / 2), int(y - height / 2),
                          int(width), int(height)])
            scores.append(score)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    start = 640 * (1 - CENTER_ZONE_FRACTION) / 2
    end = 640 - start
    for index in np.array(indices).flatten() if len(indices) else []:
        x, y, width, height = boxes[index]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(640, x + width), min(640, y + height)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        red_pixels = cv2.countNonZero(red_mask[y1:y2, x1:x2]) if area else 0
        enough_red = red_pixels >= max(RED_MIN_PIXELS, area * RED_MIN_FRACTION)
        if (start <= x + width / 2 <= end and start <= y + height / 2 <= end
                and enough_red):
            return True
    return False

pygame.init()

arduino = serial.Serial("COM12", 115200)
time.sleep(2)

WIDTH = 1200
HEIGHT = 750

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RPR Robot Simulator")

clock = pygame.time.Clock()

BG = (245, 247, 250)

BLACK = (25, 25, 25)
DARK = (40, 40, 40)

BLUE = (52, 120, 246)
BLUE_DARK = (30, 85, 190)

RED = (240, 90, 90)

GREEN = (60, 200, 120)

GRAY = (180, 185, 195)
LIGHT = (255, 255, 255)

SHADOW = (210, 214, 220)

SCALE = 14

L1 = 20
L2 = 10

rail_min = 0
rail_max = 30

rpr = RPR(L1, L2, rail_min, rail_max)

RAIL_X = 170
RAIL_Y = 620

WORKSPACE_X = RAIL_X + (12 * SCALE)
WORKSPACE_Y = RAIL_Y - (30 * SCALE)

WORKSPACE_WIDTH = (30 - 12) * SCALE
WORKSPACE_HEIGHT = 30 * SCALE

pygame.font.init()

title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
sub_font = pygame.font.SysFont("Segoe UI", 18)
value_font = pygame.font.SysFont("Consolas", 22)
small_font = pygame.font.SysFont("Segoe UI", 15)

trail = []

last_send = 0

smooth_x = 20
smooth_y = 10

MANUAL_SPEED = 3.0              # cm per second for mouse-controlled motion

smooth_shoulder = 90
smooth_elbow = 90
smooth_wrist = 90

SERVO_SPEED = 25.0              # maximum servo change, degrees per second
SERIAL_SEND_INTERVAL = 0.05     # 20 commands/sec; avoids flooding hardware

robot_armed = False
last_position = None

# Autonomous scan / camera state. The camera and model load only after I is
# pressed, so normal mouse control remains quick to start.
auto_mode = False
auto_state = "positioning"  # positioning, scanning, picking, returning
auto_rail = 0.0
auto_direction = 1
auto_arm_x = 23.0
AUTO_RAIL_SPEED = 1.5
AUTO_ARM_SPEED = 2.0
AUTO_PICK_X = 30.0
AUTO_REST_X = 23.0
AUTO_PICK_HOLD = 0.75
AUTO_SCAN_SHOULDER = 60.0
pick_until = None

camera = None
vision_session = None
vision_input_name = None
video_surface = None
vision_error = None

VIDEO_RECT = pygame.Rect(760, 480, 400, 250)
def start_vision():
    """Open the camera and load the model once; return whether it succeeded."""
    global camera, vision_session, vision_input_name, vision_error
    if camera is not None and vision_session is not None:
        return True
    try:
        vision_session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        vision_input_name = vision_session.get_inputs()[0].name
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = None
            raise RuntimeError(f"Camera {CAMERA_INDEX} not detected")
        vision_error = None
        return True
    except Exception as error:
        vision_error = str(error)
        print(f"Vision startup failed: {vision_error}")
        return False


def move_towards(current, target, maximum_step):
    """Move a numeric command toward a target without exceeding one step."""
    if current < target:
        return min(current + maximum_step, target)
    return max(current - maximum_step, target)

running = True
robot_armed = False
while running:

    delta_time = clock.tick(60) / 1000.0
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_y:

                robot_armed = not robot_armed

                if robot_armed:
                    print("ROBOT ARMED")
                    auto_mode = False

                    # Remove previous marker
                    last_position = None

                else:
                    print("ROBOT DISARMED")
                    auto_mode = False

                    # Save the last robot position
                    last_position = (
                        int(wrist_x),
                        int(wrist_y)
                    )
            elif event.key == pygame.K_i and not robot_armed:
                if auto_mode:
                    auto_mode = False
                    print("AUTONOMOUS SCAN STOPPED")
                elif start_vision():
                    auto_mode = True
                    # First move gently from the manual pose into the scan
                    # pose. The rail does not begin travelling until ready.
                    auto_state = "positioning"
                    auto_rail = smooth_y
                    auto_arm_x = smooth_x
                    pick_until = None
                    print("AUTONOMOUS SCAN STARTED")
    screen.fill(BG)

    mx, my = pygame.mouse.get_pos()
    mx = max(WORKSPACE_X, min(mx, WORKSPACE_X + WORKSPACE_WIDTH))
    my = max(WORKSPACE_Y, min(my, RAIL_Y))

    if auto_mode:
        now = time.monotonic()
        frame = None
        # Keep the Pygame video feed live in every autonomous state, including
        # while the rail is paused for an approval decision.
        if camera is not None:
            ok, frame = camera.read()
            if ok:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, VIDEO_RECT.size)
                video_surface = pygame.image.frombuffer(
                    rgb_frame.tobytes(), VIDEO_RECT.size, "RGB"
                )
        if auto_state == "positioning":
            auto_arm_x = move_towards(
                auto_arm_x, AUTO_REST_X, AUTO_ARM_SPEED * delta_time
            )
            # Let the rate-limited shoulder settle into its scan pose before
            # moving the rail.
            if (auto_arm_x == AUTO_REST_X
                    and abs(smooth_shoulder - AUTO_SCAN_SHOULDER) < 1.0):
                auto_state = "scanning"
        elif auto_state == "scanning":
            auto_rail += auto_direction * AUTO_RAIL_SPEED * delta_time
            if auto_rail >= rail_max:
                auto_rail, auto_direction = rail_max, -1
            elif auto_rail <= rail_min:
                auto_rail, auto_direction = rail_min, 1

            if frame is not None and apple_in_center(
                    vision_session, vision_input_name, frame):
                # A red, centre-zone detection is approved automatically.
                # Rail movement stops as the arm begins its rate-limited pick.
                auto_state = "picking"
                pick_until = None
                print(f"APPLE DETECTED AT RAIL {auto_rail:.1f} cm - AUTO PICK")
        elif auto_state == "picking":
            auto_arm_x = move_towards(
                auto_arm_x, AUTO_PICK_X, AUTO_ARM_SPEED * delta_time
            )
            if auto_arm_x >= AUTO_PICK_X and pick_until is None:
                pick_until = now + AUTO_PICK_HOLD
            elif pick_until is not None and now >= pick_until:
                auto_state = "returning"
        elif auto_state == "returning":
            auto_arm_x = move_towards(
                auto_arm_x, AUTO_REST_X, AUTO_ARM_SPEED * delta_time
            )
            if auto_arm_x <= AUTO_REST_X:
                auto_state = "positioning"

        x_cm, y_cm = auto_arm_x, auto_rail
        smooth_x, smooth_y = x_cm, y_cm
    elif robot_armed:
        target_x = (mx - RAIL_X) / SCALE
        target_y = (RAIL_Y - my) / SCALE
        dx, dy = target_x - smooth_x, target_y - smooth_y
        distance = math.sqrt(dx * dx + dy * dy)
        maximum_step = MANUAL_SPEED * delta_time
        if distance > maximum_step:
            dx, dy = dx / distance * maximum_step, dy / distance * maximum_step
        smooth_x += dx
        smooth_y += dy
        x_cm, y_cm = smooth_x, smooth_y
    else:
        # With autonomous mode cancelled and the robot disarmed, keep sending
        # no new motion targets. The next armed movement starts smoothly from
        # this exact held pose rather than jumping to the mouse location.
        x_cm, y_cm = smooth_x, smooth_y

    if x_cm < 12:
        x_cm = 12

    if x_cm > 30:
        x_cm = 30

    if y_cm < rail_min:
        y_cm = rail_min

    if y_cm > rail_max:
        y_cm = rail_max

    rail, elbow, shoulder = rpr.inv_kinematic(x_cm, y_cm)

    # During autonomous scanning, keep the shoulder in its 60-degree viewing
    # pose. The elbow remains the normal inverse-kinematics value for the
    # current rail position. smooth_shoulder below ramps the physical command
    # to this target rather than jumping to it.
    if auto_mode and auto_state in ("positioning", "scanning"):
        shoulder = AUTO_SCAN_SHOULDER

    wrist = 270 - elbow - shoulder

    servo_step = SERVO_SPEED * delta_time
    smooth_shoulder = move_towards(smooth_shoulder, shoulder, servo_step)
    smooth_elbow = move_towards(smooth_elbow, elbow, servo_step)
    smooth_wrist = move_towards(smooth_wrist, wrist, servo_step)

    shoulder_rad = math.radians(smooth_shoulder)

    elbow_internal = smooth_elbow - 180
    elbow_rad = math.radians(elbow_internal)

    rail_px = RAIL_Y - rail * SCALE

    shoulder_x = RAIL_X
    shoulder_y = rail_px

    elbow_x = shoulder_x + (L1 * SCALE) * math.cos(shoulder_rad)
    elbow_y = shoulder_y - (L1 * SCALE) * math.sin(shoulder_rad)

    wrist_x = elbow_x + (L2 * SCALE) * math.cos(shoulder_rad + elbow_rad)
    wrist_y = elbow_y - (L2 * SCALE) * math.sin(shoulder_rad + elbow_rad)

    trail.append((int(wrist_x), int(wrist_y)))

    if len(trail) > 150:
        trail.pop(0)

    current_time = time.time()

    if (robot_armed or auto_mode) and current_time - last_send > SERIAL_SEND_INTERVAL:

        data = (
            f"R{rail:.1f},"
            f"S{int(smooth_shoulder)},"
            f"E{int(smooth_elbow)},"
            f"W{int(smooth_wrist)}\n"
        )

        arduino.write(data.encode())

        last_send = current_time

    for i in range(1, len(trail)):

        pygame.draw.line(
            screen,
            (180, 210, 255),
            trail[i - 1],
            trail[i],
            2
        )

    workspace_rect = pygame.Rect(
        WORKSPACE_X,
        WORKSPACE_Y,
        WORKSPACE_WIDTH,
        WORKSPACE_HEIGHT
    )

    pygame.draw.rect(
        screen,
        (225, 232, 245),
        workspace_rect,
        border_radius=10
    )

    pygame.draw.rect(
        screen,
        BLUE,
        workspace_rect,
        3,
        border_radius=10
    )

    pygame.draw.rect(
        screen,
        SHADOW,
        (
            RAIL_X - 35,
            RAIL_Y - rail_max * SCALE - 10,
            70,
            rail_max * SCALE + 50
        ),
        border_radius=14
    )

    pygame.draw.rect(
        screen,
        DARK,
        (
            RAIL_X - 28,
            RAIL_Y - rail_max * SCALE,
            56,
            rail_max * SCALE + 35
        ),
        border_radius=12
    )

    pygame.draw.line(
        screen,
        LIGHT,
        (RAIL_X, RAIL_Y),
        (RAIL_X, RAIL_Y - rail_max * SCALE),
        8
    )

    pygame.draw.rect(
        screen,
        BLUE,
        (
            shoulder_x - 30,
            shoulder_y - 30,
            60,
            60
        ),
        border_radius=10
    )

    pygame.draw.circle(
        screen,
        BLUE_DARK,
        (int(shoulder_x), int(shoulder_y)),
        12
    )

    pygame.draw.line(
        screen,
        BLACK,
        (shoulder_x, shoulder_y),
        (elbow_x, elbow_y),
        14
    )

    pygame.draw.line(
        screen,
        BLUE_DARK,
        (shoulder_x, shoulder_y),
        (elbow_x, elbow_y),
        6
    )

    pygame.draw.line(
        screen,
        BLACK,
        (elbow_x, elbow_y),
        (wrist_x, wrist_y),
        12
    )

    pygame.draw.line(
        screen,
        BLUE,
        (elbow_x, elbow_y),
        (wrist_x, wrist_y),
        5
    )

    pygame.draw.circle(
        screen,
        BLUE_DARK,
        (int(elbow_x), int(elbow_y)),
        15
    )

    pygame.draw.circle(
        screen,
        LIGHT,
        (int(elbow_x), int(elbow_y)),
        5
    )

    pygame.draw.circle(
        screen,
        RED,
        (int(wrist_x), int(wrist_y)),
        10
    )

    pygame.draw.line(
        screen,
        RED,
        (wrist_x, wrist_y),
        (wrist_x, wrist_y + 30),
        3
    )

    pygame.draw.circle(
        screen,
        GREEN,
        (mx, my),
        6
    )

    pygame.draw.circle(
        screen,
        (255, 255, 255),
        (mx, my),
        14,
        2
    )

    panel_x = 760
    panel_y = 55

    pygame.draw.rect(
        screen,
        SHADOW,
        (
            panel_x - 8,
            panel_y - 8,
            320,
            360
        ),
        border_radius=16
    )

    pygame.draw.rect(
        screen,
        LIGHT,
        (
            panel_x - 15,
            panel_y - 15,
            320,
            360
        ),
        border_radius=16
    )

    pygame.draw.rect(
        screen,
        GRAY,
        (
            panel_x - 15,
            panel_y - 15,
            320,
            360
        ),
        2,
        border_radius=16
    )

    title = title_font.render("RPR Robot", True, BLACK)

    screen.blit(title, (panel_x + 15, panel_y))

    subtitle = sub_font.render(
        "Live Hardware Control",
        True,
        (110, 110, 110)
    )

    screen.blit(subtitle, (panel_x + 15, panel_y + 40))

    values = [
        ("X Position", f"{round(x_cm,1)} cm"),
        ("Rail Height", f"{round(rail,1)} cm"),
        ("Shoulder", f"{int(smooth_shoulder)} deg"),
        ("Elbow", f"{int(smooth_elbow)} deg"),
        ("Wrist", f"{int(smooth_wrist)} deg"),
        # ("Serial", "CONNECTED")
    ]

    text_y = panel_y + 95

    for label, value in values:

        pygame.draw.rect(
            screen,
            (248, 250, 252),
            (
                panel_x + 10,
                text_y - 8,
                260,
                36
            ),
            border_radius=8
        )

        label_surface = small_font.render(
            label,
            True,
            (90, 90, 90)
        )

        value_surface = value_font.render(
            value,
            True,
            BLUE_DARK
        )

        screen.blit(label_surface, (panel_x + 18, text_y - 1))
        screen.blit(value_surface, (panel_x + 140, text_y - 6))

        text_y += 46

    footer = small_font.render(
        "Mouse controls real robot",
        True,
        (120, 120, 120)
    )

    screen.blit(footer, (30, HEIGHT - 30))
    if robot_armed:
        status_text = "ARMED - MOUSE CONTROL ACTIVE"
        status_color = GREEN
    elif auto_mode:
        status_text = "DISARMED - AUTO SCAN ACTIVE"
        status_color = BLUE_DARK
    else:
        status_text = "DISARMED - PRESS Y TO ENABLE"
        status_color = RED

    status_surface = sub_font.render(
        status_text,
        True,
        status_color
    )

    screen.blit(status_surface, (760, 440))

    # Camera panel is intentionally inside the Pygame GUI (no separate OpenCV
    # window). It becomes live after Y, then I, starts autonomous scanning.
    pygame.draw.rect(screen, DARK, VIDEO_RECT.inflate(8, 8), border_radius=8)
    if video_surface is not None:
        screen.blit(video_surface, VIDEO_RECT)
        zone_width = int(VIDEO_RECT.width * CENTER_ZONE_FRACTION)
        zone_height = int(VIDEO_RECT.height * CENTER_ZONE_FRACTION)
        zone = pygame.Rect(
            VIDEO_RECT.centerx - zone_width // 2,
            VIDEO_RECT.centery - zone_height // 2,
            zone_width,
            zone_height,
        )
        pygame.draw.rect(screen, (255, 220, 0), zone, 2)
        zone_text = small_font.render("DETECTION ZONE", True, (255, 220, 0))
        screen.blit(zone_text, (zone.x, zone.y - 18))
    else:
        camera_text = small_font.render(
            "Disarm with Y, then press I to scan", True, LIGHT
        )
        screen.blit(camera_text, (VIDEO_RECT.x + 25, VIDEO_RECT.y + 110))

    if auto_mode:
        auto_labels = {
            "positioning": "AUTO: MOVING TO SCAN POSE",
            "scanning": "AUTO: SCANNING RAIL (I to stop)",
            "picking": "AUTO: MOVING ARM TO X = 30",
            "returning": "AUTO: RETURNING ARM",
        }
        auto_text = small_font.render(auto_labels[auto_state], True, BLUE_DARK)
        screen.blit(auto_text, (VIDEO_RECT.x, VIDEO_RECT.y - 25))
    # ==========================================
    # LAST POSITION MARKER
    # ==========================================

    if not robot_armed and last_position is not None:
        # Outer circle
        pygame.draw.circle(
            screen,
            RED,
            last_position,
            12
        )

        # Inner circle
        pygame.draw.circle(
            screen,
            LIGHT,
            last_position,
            5
        )

        # Crosshair
        pygame.draw.line(
            screen,
            RED,
            (last_position[0] - 18, last_position[1]),
            (last_position[0] + 18, last_position[1]),
            2
        )

        pygame.draw.line(
            screen,
            RED,
            (last_position[0], last_position[1] - 18),
            (last_position[0], last_position[1] + 18),
            2
        )

        marker_text = small_font.render(
            "LAST POSITION",
            True,
            RED
        )

        screen.blit(
            marker_text,
            (last_position[0] + 18, last_position[1] - 25)
        )
    pygame.display.update()

arduino.close()

if camera is not None:
    camera.release()

pygame.quit()
