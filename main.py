import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import json
import os
import socket
import tkinter as tk
from tkinter import ttk
import threading

# --- Настройки ---
SERVER_IP = "127.0.0.1"  # Укажи IP сервера
SERVER_PORT = 8080
PITCH_THRESHOLD = 0.4
YAW_THRESHOLD = 0.4
EYE_LOOK_THRESHOLD = 0.2
MAX_AWAY_DURATION = 3.0
LOG_FILE = "exam_events.json"
MESH_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

# --- Глобальные переменные ---
selected_camera_index = 0
stop_client_flag = False
log_events = []

def select_camera():
    global selected_camera_index
    def on_select():
        global selected_camera_index
        selected_camera_index = int(combo_cameras.get().split()[1])
        root.destroy()

    root = tk.Tk()
    root.title("Выбор камеры")

    label = tk.Label(root, text="Выберите камеру:")
    label.pack(pady=10)

    cap_id = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(cap_id)
        if cap.isOpened():
            cameras.append(f"Камера {cap_id}")
            cap.release()
        else:
            break
        cap_id += 1

    combo_cameras = ttk.Combobox(root, values=cameras)
    if cameras:
        combo_cameras.current(0)
    combo_cameras.pack(pady=10)

    button_ok = tk.Button(root, text="OK", command=on_select)
    button_ok.pack(pady=10)

    root.mainloop()

def draw_russian_text(img, text, position, font_size=20, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = 'DejaVuSans.ttf'
    if os.name == 'nt':
        font_path = os.path.join('C:', 'Windows', 'Fonts', 'arial.ttf')
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, FileNotFoundError):
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def estimate_head_pose(landmarks, image_shape):
    h, w = image_shape[:2]
    face_2d = []
    face_3d = []
    for idx in [1, 33, 263, 152, 50, 280]:
        if idx < len(landmarks):
            x = landmarks[idx].x
            y = landmarks[idx].y
            z = landmarks[idx].z
            face_2d.append([x * w, y * h])
            face_3d.append([x * w, y * h, z * w])

    if len(face_2d) < 4:
        return None, None, None

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, w / 2],
                           [0, focal_length, h / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll

def estimate_gaze(landmarks, image_shape):
    h, w = image_shape[:2]
    try:
        left_eye_inner = landmarks[33]
        left_eye_outer = landmarks[133]
        right_eye_inner = landmarks[362]
        right_eye_outer = landmarks[263]

        lecx = (left_eye_inner.x + left_eye_outer.x) / 2
        recx = (right_eye_inner.x + right_eye_outer.x) / 2
        ecx = (lecx + recx) / 2
        nx = landmarks[1].x

        offset = ecx - nx
        ew = abs(left_eye_outer.x - left_eye_inner.x)
        if ew > 0.01:
            no = offset / ew
            return abs(no) > EYE_LOOK_THRESHOLD
    except (IndexError, AttributeError):
        pass
    return False

def send_alert(data):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_IP, SERVER_PORT))
        s.send(json.dumps(data).encode('utf-8'))
        s.close()
    except Exception as e:
        print(f"Ошибка отправки: {e}")

last_event_times = {}
def log_and_send(event_type, ts, details=""):
    key = event_type
    if key in last_event_times and ts - last_event_times[key] < 5:
        return
    last_event_times[key] = ts

    entry = {"time_sec": round(ts, 2), "event": event_type, "info": details}
    log_events.append(entry)
    print(f"[{ts:.2f}s] {event_type}: {details}")

    thread = threading.Thread(target=send_alert, args=(entry,))
    thread.daemon = True
    thread.start()

def main():
    global stop_client_flag
    select_camera()

    cap = cv2.VideoCapture(selected_camera_index)
    if not cap.isOpened():
        print("Ошибка: камера не найдена.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    start_time = time.time()
    last_away_start = None
    was_looking_away = False
    multi_face_logged = False

    while True:
        if stop_client_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mesh_results = face_mesh.process(rgb)

        elapsed = time.time() - start_time

        # Подсчёт лиц только по FaceMesh
        num_faces_mesh = len(mesh_results.multi_face_landmarks) if mesh_results.multi_face_landmarks else 0

        if num_faces_mesh == 0:
            frame = draw_russian_text(frame, "Нет лица", (10, 30), font_size=18, color=(0, 0, 255))
        elif num_faces_mesh > 1:
            frame = draw_russian_text(frame, "Много лиц!", (10, 30), font_size=18, color=(0, 0, 255))
            if not multi_face_logged:
                log_and_send("MULTIPLE_FACES", elapsed, f"{num_faces_mesh} лиц")
                multi_face_logged = True
        else:
            multi_face_logged = False

        current_looking_away = False
        if mesh_results.multi_face_landmarks and num_faces_mesh >= 1:
            # Работаем с первым лицом
            lm = mesh_results.multi_face_landmarks[0].landmark

            # Рисуем сетку для первого лица
            ih, iw, ic = frame.shape
            for connection in MESH_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(lm) and end_idx < len(lm):
                    start_point = (int(lm[start_idx].x * iw), int(lm[start_idx].y * ih))
                    end_point = (int(lm[end_idx].x * iw), int(lm[end_idx].y * ih))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 1)

            # Проверка поворота головы
            pitch, yaw, _ = estimate_head_pose(lm, frame.shape)
            if pitch is not None and yaw is not None:
                if abs(pitch) > PITCH_THRESHOLD or abs(yaw) > YAW_THRESHOLD:
                    current_looking_away = True

            # Проверка взгляда
            if estimate_gaze(lm, frame.shape):
                current_looking_away = True

        # Логика отслеживания отсутствия взгляда
        if current_looking_away:
            if not was_looking_away:
                last_away_start = elapsed
                was_looking_away = True
        else:
            if was_looking_away:
                if last_away_start and elapsed - last_away_start > MAX_AWAY_DURATION:
                    log_and_send("LOOKING_AWAY", elapsed, f"Длительность: {elapsed - last_away_start:.1f} с")
                was_looking_away = False

        # Отображение статуса
        if current_looking_away and was_looking_away and last_away_start and elapsed - last_away_start <= MAX_AWAY_DURATION:
            frame = draw_russian_text(frame, "Смотрит в сторону", (10, 60), font_size=18, color=(0, 69, 255))

        frame = draw_russian_text(frame, f"Время: {elapsed:.1f} с", (10, 450), font_size=16, color=(255, 255, 255))
        cv2.imshow('Контроль', frame)

        if cv2.waitKey(1) == 27:  # ESC
            stop_client_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_events, f, indent=2, ensure_ascii=False)
    print(f"Лог сохранён в '{LOG_FILE}'")

if __name__ == "__main__":
    main()