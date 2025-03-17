import cv2
import mediapipe as mp
import numpy as np

# אתחול MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# פונקציה לחישוב זווית בין שלוש נקודות
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# פונקציה לזיהוי תנוחת עץ
def detect_tree_pose(landmarks):
    left_knee_angle = calculate_angle(
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
        (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
        (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
    )

    hands_above_head = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.NOSE.value].y and \
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.NOSE.value].y

    return left_knee_angle > 100 and hands_above_head  # תנאי לזיהוי תנוחת עץ

# פונקציה לזיהוי תנוחת פלאנק
def detect_plank_pose(landmarks):
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    body_angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (ankle.x, ankle.y))

    return 160 < body_angle < 180  # גוף בקו ישר

# פונקציה לזיהוי תנוחת סירה
def detect_boat_pose(landmarks):
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    knee_angle = calculate_angle((left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y))

    hands_up = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y

    return 130 < knee_angle < 160 and hands_up  # רגליים בזווית בינונית וידיים מורמות

# פונקציה לזיהוי תנוחת משולש
def detect_triangle_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    upper_body_angle = calculate_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_ankle.x, left_ankle.y))

    one_hand_up = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

    return 50 < upper_body_angle < 80 and one_hand_up  # גוף נוטה הצידה

# פונקציה לזיהוי תנוחת עמידה (Tadasana)
def detect_standing_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    body_angle = calculate_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_ankle.x, left_ankle.y))

    feet_together = abs(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x) < 0.05
    hands_at_sides = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y - landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y) < 0.1 and \
                     abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) < 0.1

    return 170 < body_angle < 180 and feet_together and hands_at_sides  # תנוחה יציבה עם רגליים צמודות וידיים לצידי הגוף

# פונקציה ראשית להרצת זיהוי התנוחות
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            detected_pose = None
            if detect_standing_pose(landmarks):
                detected_pose = "Standing Pose"
            elif detect_tree_pose(landmarks):
                detected_pose = "Tree Pose"
            elif detect_plank_pose(landmarks):
                detected_pose = "Plank Pose"
            elif detect_boat_pose(landmarks):
                detected_pose = "Boat Pose"
            elif detect_triangle_pose(landmarks):
                detected_pose = "Triangle Pose"

            if detected_pose:
                cv2.putText(frame, f"{detected_pose} Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Yoga Pose Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
