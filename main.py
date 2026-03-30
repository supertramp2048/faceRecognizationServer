from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import traceback


app = Flask(__name__)

# ================= MODEL =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

detector = cv2.FaceDetectorYN.create(
    os.path.join(BASE_DIR, "models/face_detection_yunet_2023mar.onnx"),
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF.create(
    os.path.join(BASE_DIR, "models/face_recognition_sface_2021dec.onnx"),
    ""
)

# ================= DATABASE =================
known_faces = []
known_names = []

DB_PATH = "faces"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# ================= LOAD =================
def load_faces():
    global known_faces, known_names

    known_faces = []
    known_names = []

    for file in os.listdir(DB_PATH):
        path = os.path.join(DB_PATH, file)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w, _ = img.shape
        detector.setInputSize((w, h))

        _, faces = detector.detect(img)

        if faces is not None:
            face = faces[0]
            aligned = recognizer.alignCrop(img, face)
            feature = recognizer.feature(aligned)

            known_faces.append(feature)
            known_names.append(file.split('.')[0])

    print("Loaded:", known_names)

load_faces()

# ================= REGISTER =================
@app.route('/register', methods=['POST'])
def register():
    print("\n--- BAT DAU NHAN REQUEST ---")
    try:
        name = request.args.get("name")
        print(f"1. Ten nguoi dung: {name}")

        if not name:
            return "Missing name"

        data = request.get_data()
        print(f"2. Dung luong anh nhan duoc: {len(data)} bytes")

        if len(data) == 0:
            print(">> LOI: Postman khong gui duoc byte du lieu nao!")
            return "No data received"

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(">> LOI: OpenCV khong the doc duoc cau truc anh nay!")
            return "Invalid image format"
            
        print("3. OpenCV da doc anh thanh cong")

        h, w, _ = img.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None:
            print(">> LOI: Khong tim thay khuon mat trong anh!")
            return "No face detected"
            
        print("4. Da phat hien duoc khuon mat")

        path = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(path, img)
        print(f"5. Da luu file anh vao: {path}")

        load_faces()
        print("6. Tai lai Database thanh cong!")

        return "OK"

    except Exception as e:
        print("=== LOI BEN TRONG TRY-CATCH ===")
        print(str(e))
        return "FAIL"

# ================= RECOGNITION =================
@app.route('/check_face_upload', methods=['POST'])
def upload():
    try:
        data = request.data
        
        # 1. Chặn lỗi 400 Bad Request (ESP32 gửi gói tin rỗng)
        if not data:
            print(">> LOI: Khong nhan duoc du lieu anh!")
            return "FAIL", 200

        # 2. Giải mã ảnh JPEG
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(">> LOI: Anh bi hong hoac khong the decode!")
            return "FAIL", 200

        # Lưu ảnh mới nhất để debug
        cv2.imwrite("debug.jpg", img) 
        
        h, w, _ = img.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        # 3. Kịch bản CÓ KHUÔN MẶT
        if faces is not None:
            face = faces[0]
            aligned = recognizer.alignCrop(img, face)
            feature = recognizer.feature(aligned)

            best_score = 0
            best_name = "Unknown"

            for i, db_feature in enumerate(known_faces):
                score = recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = known_names[i]

            print(f"Quet xong -> Score cao nhat: {best_score:.4f} ({best_name})")

            # Quyết định Mở hay Khóa
            if best_score > 0.5:
                print(f"==> MATCH: {best_name} -> Gui lenh [OPEN]")
                return "SUCCESS", 200  # ESP32 dang cho chu nay
            else:
                print("==> SCORE THAP -> Gui lenh [DENY]")
                return "FAIL", 200  # ESP32 dang cho chu nay

        # 4. Kịch bản KHÔNG THẤY MẶT (Góc khuất, quá tối...)
        else:
            print(">> Khong tim thay khuon mat nao trong buc anh -> Gui lenh [DENY]")
            return "FAIL", 200

    except Exception as e:
        print("=== LOI SERVER KHI RECOGNITION ===")
        print(traceback.format_exc()) # In ra chi tiet dong bi loi
        return "FAIL", 200
# ================= LIST USERS =================
@app.route('/list', methods=['GET'])
def list_users():
    return jsonify(known_names)

# ================= RUN =================
app.run(host="0.0.0.0", port=5001, debug=False)