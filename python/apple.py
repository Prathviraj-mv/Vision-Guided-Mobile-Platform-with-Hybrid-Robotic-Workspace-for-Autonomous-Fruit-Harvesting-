
import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
model_path = "best_model2.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

# Get model input information
input_info = session.get_inputs()[0]
input_name = input_info.name

print("Model input:", input_info.shape)

# Model input dimensions
input_height = 640
input_width = 640

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not detected")
    exit()

CONF_THRESHOLD = 0.8
IOU_THRESHOLD = 0.6

while True:

    ret, frame = cap.read()

    if not ret:
        break

    original_h, original_w = frame.shape[:2]

    # Resize and prepare input
    image = cv2.resize(frame, (input_width, input_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    # ONNX inference
    outputs = session.run(
        None,
        {input_name: image}
    )

    predictions = outputs[0]

    print("Output shape:", predictions.shape) if False else None

    # YOLOv8 output: [1, 4 + classes, candidates]
    predictions = np.squeeze(predictions)

    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []

    for detection in predictions:

        if len(detection) < 5:
            continue

        x, y, w, h = detection[:4]

        confidence = float(np.max(detection[4:]))

        if confidence < CONF_THRESHOLD:
            continue

        # Convert center coordinates to corners
        x1 = int((x - w / 2) * original_w / input_width)
        y1 = int((y - h / 2) * original_h / input_height)

        x2 = int((x + w / 2) * original_w / input_width)
        y2 = int((y + h / 2) * original_h / input_height)

        boxes.append([
            x1, y1, x2 - x1, y2 - y1
        ])

        scores.append(confidence)

    # Remove duplicate detections
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    apple_count = 0

    if len(indices) > 0:

        for i in np.array(indices).flatten():

            x, y, w, h = boxes[i]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Apple {scores[i]:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            apple_count += 1

    cv2.putText(
        frame,
        f"Apples: {apple_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Apple Detection - ONNX", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
