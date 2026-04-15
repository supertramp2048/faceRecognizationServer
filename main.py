from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
import numpy as np
import os
import sys
import collections
import socket
import mediapipe as mp
import math
import requests
import json
import threading
from dotenv import load_dotenv
from datetime import datetime
from functools import wraps

# Load .env từ thư mục hiện tại của script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

app = Flask(__name__)
# Khôi phục dùng thư viện CORS để tự động xử lý preflight OPTIONS
CORS(app, resources={r"/*": {"origins": "*"}})

# ================= CẤU HÌNH THINGSBOARD =================
THINGSBOARD_SERVER = "demo.thingsboard.io"

def verify_token_with_thingsboard(token):
    """
    Verify Bearer token từ ThingsBoard
    Gọi endpoint ThingsBoard để kiểm tra token có hợp lệ không
    """
    try:
        print(f">> Verifying token: {token[:20]}...") # Debug
        
        verify_url = f"https://{THINGSBOARD_SERVER}/api/auth/user"
        
        response = requests.get(verify_url, headers={'Authorization': f'Bearer {token}'})
        
        print(f">> ThingsBoard response status: {response.status_code}") # Debug
        
        if response.status_code == 200:
            # Token hợp lệ, trả về user info
            print(f">> Token hợp lệ - User: {response.json()}") # Debug
            return True, response.json()
        else:
            # Token không hợp lệ
            print(f">> Token không hợp lệ - Response: {response.text}") # Debug
            return False, None
    except Exception as e:
        print(f">> Loi khi verify token: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def token_required(f):
    """
    Decorator để kiểm tra Bearer token trong header
    Sử dụng: @token_required
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip token verify cho CORS preflight OPTIONS request
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        
        print(f">> Request headers: {dict(request.headers)}") # Debug
        
        token = None
        
        # Kiểm tra Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            print(f">> Authorization header: {auth_header}") # Debug
            try:
                # Format: "Bearer <token>"
                token = auth_header.split(" ")[1]
                print(f">> Extracted token: {token[:20]}...") # Debug
            except IndexError:
                print(">> ERROR: Invalid authorization header format") # Debug
                return jsonify({"status": "error", "message": "Invalid authorization header format"}), 401
        else:
            print(">> ERROR: Authorization header not found") # Debug
        
        if not token:
            return jsonify({"status": "error", "message": "Token is missing"}), 401
        
        # Verify token với ThingsBoard
        is_valid, user_info = verify_token_with_thingsboard(token)
        
        if not is_valid:
            return jsonify({"status": "error", "message": "Invalid or expired token"}), 401
        
        # Lưu thông tin user vào request context để dùng trong function
        request.user_info = user_info
        request.access_token = token
        
        return f(*args, **kwargs)
    
    return decorated_function

def send_to_thingsboard(person_name):
    ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
    url = f"https://{THINGSBOARD_SERVER}/api/v1/{ACCESS_TOKEN}/telemetry"
    headers = {'Content-Type': 'application/json'}
    
    # Timestamp theo milliseconds (như ThingsBoard yêu cầu)
    current_timestamp = int(datetime.now().timestamp() * 1000)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    payload = {
        "door": True,
        "person_name": person_name,
        "open_time": current_time,
        "ts": current_timestamp
    }
    
    try:
        print(f"DEBUG - Sending to: {url[:60]}...")
        print(f"DEBUG - Payload: {payload}")
        
        # Dùng json= thay vì data=json.dumps()
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        print(f"DEBUG - Status: {response.status_code}")
        print(f"DEBUG - Response: {response.text}")
        
        if response.status_code == 200:
            print(f">> Da gui du lieu len ThingsBoard: {payload}")
        else:
            print(f">> Loi ThingsBoard: {response.status_code} - {response.text}")
    except Exception as e:
        print(f">> Loi ket noi ThingsBoard: {e}")
        import traceback
        traceback.print_exc()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
sys.stdout.reconfigure(encoding='utf-8')

# ================= ĐỊA CHỈ ESP32-CAM =================
ESP32_STREAM_URL = os.getenv('ESP32_STREAM_URL') 

def get_esp32_url():
        return ESP32_STREAM_URL
    
# ================= KHỞI TẠO MODEL AI =================
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

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye_points):
    d1 = math.hypot(eye_points[1][0] - eye_points[5][0], eye_points[1][1] - eye_points[5][1])
    d2 = math.hypot(eye_points[2][0] - eye_points[4][0], eye_points[2][1] - eye_points[4][1])
    d3 = math.hypot(eye_points[0][0] - eye_points[3][0], eye_points[0][1] - eye_points[3][1])
    
    if d3 == 0: 
        return 0
    return (d1 + d2) / (2.0 * d3)

# ================= CƠ SỞ DỮ LIỆU =================
known_faces = []
known_names = []
db_lock = threading.Lock() 

DB_PATH = os.path.join(BASE_DIR, "faces")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "faces_db.npz")

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def classify_head_pose(face):
    right_eye_x = face[4]
    left_eye_x = face[6]
    nose_x = face[8]
    
    dist_to_right = nose_x - right_eye_x
    dist_to_left = left_eye_x - nose_x
    total_dist = dist_to_right + dist_to_left
    
    if total_dist == 0: 
        return "front"
    
    ratio = dist_to_right / total_dist
    
    if ratio < 0.40:
        return "left"   
    elif ratio > 0.60:
        return "right"  
    else:
        return "front"  

def build_and_save_vectors():
    print("\n--- DANG XAY DUNG DATABASE VECTOR 3 GOC MAT ---")
    person_features = {}

    for file in os.listdir(DB_PATH):
        if not file.endswith(('.jpg', '.png')): continue
            
        path = os.path.join(DB_PATH, file)
        img = cv2.imread(path)
        if img is None: continue

        h, w, _ = img.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is not None:
            face = faces[0] 
            pose = classify_head_pose(face)
            
            aligned = recognizer.alignCrop(img, face) 
            feature = recognizer.feature(aligned) 
            
            name = file.split('_')[0].split('.')[0]
            
            if name not in person_features:
                person_features[name] = {'front': [], 'left': [], 'right': []}
                
            person_features[name][pose].append(feature[0]) 

    final_names = []
    final_features = []

    for name, poses in person_features.items():
        for pose, features_list in poses.items():
            if len(features_list) > 0: 
                avg_feature = np.mean(features_list, axis=0)
                norm_feature = avg_feature / np.linalg.norm(avg_feature)
                norm_feature = np.expand_dims(norm_feature, axis=0).astype(np.float32)
                
                final_names.append(f"{name}_{pose}")
                final_features.append(norm_feature)

    with db_lock:
        np.savez(VECTOR_DB_PATH, names=final_names, features=final_features)
    print(f">> Da tao xong {len(final_names)} vector dai dien (Bao gom cac goc). Luu vao {VECTOR_DB_PATH}\n")


def load_faces():
    global known_faces, known_names
    
    if not os.path.exists(VECTOR_DB_PATH):
        build_and_save_vectors()

    with db_lock:
        with np.load(VECTOR_DB_PATH) as data: 
            known_names = data['names'].tolist()
            known_faces = [f for f in data['features']]
    
    unique_base_names = list(set([n.split('_')[0] for n in known_names]))
    
    print("\n--- DA TAI DU LIEU VECTOR KHUON MAT ---")
    print("Danh sach nguoi dung (Goc):", unique_base_names)
    print(f"Tong so vector goc do luu tru: {len(known_faces)}\n")

load_faces()

# ================= NHẬN DIỆN TỪ STREAM =================
@app.route('/trigger_stream', methods=['GET'])
def trigger_stream():
    print("\n" + "="*50)
    print("BAT DAU TRUY CAP STREAM & NHAN DIEN...")
    print("="*50)
    global cached_esp_ip
    url = get_esp32_url()

    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(">> LOI: Khong the mo luong stream tu ESP32-CAM!")
        cached_esp_ip = None
        url = get_esp32_url()
        cap = cv2.VideoCapture(url) 
        if not cap.isOpened():
             return "FAIL", 200

    print(">> Dang quet camera de tim NGUOI QUEN...")
    
    max_frames = 150 
    frame_count = 0
    match_found = False
    
    buffer_size = 5    
    min_votes = 3      
    recent_faces = []  
    
    # Thêm biến trạng thái để chia luồng
    identity_confirmed = False
    confirmed_name = ""

    while frame_count < max_frames:
        ret, img = cap.read()
        if not ret:
            print(">> LOI: Mat ket noi stream giua chung!")
            break
            
        frame_count += 1
        h_img, w_img, _ = img.shape
        
        # BƯỚC 1: KIỂM TRA VÀ XÁC NHẬN KHUÔN MẶT TRƯỚC
        if not identity_confirmed:
            cv2.putText(img, "DANG NHAN DIEN...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            detector.setInputSize((w_img, h_img))
            _, faces = detector.detect(img)
            
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

                if best_score > 0.45:
                    real_name = best_name.split('_')[0] 
                    print(f"  [Frame {frame_count}] Nhan dien tam thoi: {real_name} (Score: {best_score:.4f})")
                    recent_faces.append(real_name)
                else:
                    print(f"  [Frame {frame_count}] Phat hien nguoi la hoac hinh anh mo (Score: {best_score:.4f})")
                    recent_faces.append("Unknown")

                if len(recent_faces) > buffer_size:
                    recent_faces.pop(0)

                if len(recent_faces) == buffer_size:
                    vote_counts = collections.Counter(recent_faces)
                    most_common_name, most_common_count = vote_counts.most_common(1)[0]

                    if most_common_name != "Unknown" and most_common_count >= min_votes:
                        print(f"\n>> [XAC NHAN DANH TINH] ==> [{most_common_name}] voi {most_common_count}/{buffer_size} phieu bau. Chuyen sang kiem tra chop mat!")
                        identity_confirmed = True
                        confirmed_name = most_common_name

            cv2.imshow("ESP32-CAM Stream", img)
            cv2.waitKey(1)
            
        # BƯỚC 2: KIỂM TRA CHỚP MẮT (CHỈ CHẠY KHI ĐÃ XÁC NHẬN LÀ NGƯỜI QUEN)
        else:
            cv2.putText(img, f"CHAO {confirmed_name.upper()}! HAY CHOP MAT DE XAC NHAN!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                left_eye = [(int(landmarks[i].x * w_img), int(landmarks[i].y * h_img)) for i in LEFT_EYE_IDX]
                right_eye = [(int(landmarks[i].x * w_img), int(landmarks[i].y * h_img)) for i in RIGHT_EYE_IDX]
                
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < 0.22:
                    print(f">> [Frame {frame_count}] DA PHAT HIEN CHOP MAT (EAR: {avg_ear:.2f}) -> Nguoi that!")
                    print(f"\n>> [KET QUA CUOI CUNG] ==> MO CUA CHO: [{confirmed_name}]")
                    match_found = True
                    send_to_thingsboard(confirmed_name)
                    break 

            cv2.imshow("ESP32-CAM Stream", img)
            cv2.waitKey(1)

    if not match_found:
        if identity_confirmed:
            print(f"\n>> DA QUET {frame_count} FRAME: Nhan dien duoc {confirmed_name} nhung KHONG vuot qua bai test chop mat.")
        else:
            print(f"\n>> DA QUET {frame_count} FRAME: Khong the xac nhan nguoi quen.")

    cap.release()
    cv2.destroyAllWindows() 
    
    print("--- DA DONG KET NOI STREAM ---")
    
    if match_found:
        return "SUCCESS", 200
    else:
        return "FAIL", 200

# ================= ĐĂNG KÝ NGƯỜI DÙNG MỚI QUA STREAM =================
@app.route('/register', methods=['GET', 'POST', 'OPTIONS'])
@token_required
def register():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    print("\n" + "="*50)
    print("--- BAT DAU DANG KY KHUON MAT MOI TU STREAM ---")
    print("="*50)
    try:
        name = request.args.get("name")
        print(f"1. Ten dang ky: {name}")

        if not name:
            return jsonify({"status": "error", "message": "Thieu tham so name"}), 400

        db_names = []
        db_features = []
        
        with db_lock:
            if os.path.exists(VECTOR_DB_PATH):
                with np.load(VECTOR_DB_PATH) as data: 
                    db_names = data['names']
                    db_features = data['features']

        print(f">> Dang mo stream de lay anh dang ky...")
        global cached_esp_ip
        url = get_esp32_url()
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print(">> LOI: Khong the mo luong stream tu ESP32-CAM!")
            cached_esp_ip = None
            url = get_esp32_url()
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Khong the mo camera"}), 500
            
        max_frames = 200
        frame_count = 0
        captured_count = 0
        target_images = 5
        is_checked_existence = False 
        has_blinked = False

        print(f">> Vui long nhin vao camera. Dang tien hanh kiem tra va chup {target_images} buc anh...")

        while frame_count < max_frames and captured_count < target_images:
            ret, img = cap.read()
            if not ret:
                print(">> LOI: Mat ket noi stream giua chung!")
                break
                
            frame_count += 1
            h_img, w_img, _ = img.shape
            
            if not has_blinked:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_img)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_eye = [(int(landmarks[i].x * w_img), int(landmarks[i].y * h_img)) for i in LEFT_EYE_IDX]
                    right_eye = [(int(landmarks[i].x * w_img), int(landmarks[i].y * h_img)) for i in RIGHT_EYE_IDX]
                    
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    if avg_ear < 0.22:
                        has_blinked = True
                        print(f">> DA PHAT HIEN CHOP MAT (EAR: {avg_ear:.2f}). Bat dau chup anh...")

                if not has_blinked:
                    cv2.putText(img, "HAY CHOP MAT DE BAT DAU DANG KY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("Dang ky khuon mat", img)
                    cv2.waitKey(1)
                    continue
            
            cv2.imshow("Dang ky khuon mat", img)
            cv2.waitKey(1)

            detector.setInputSize((w_img, h_img))
            _, faces = detector.detect(img)

            if faces is not None:
                face = faces[0]
                
                if not is_checked_existence:
                    aligned = recognizer.alignCrop(img, face)
                    new_feature = recognizer.feature(aligned)[0]
                    new_feature = new_feature / np.linalg.norm(new_feature) 

                    if len(db_features) > 0:
                        similarities = np.dot(db_features, new_feature)
                        best_match_index = np.argmax(similarities)
                        best_score = float(np.max(similarities))

                        if best_score >= 0.4: 
                            matched_name = db_names[best_match_index].split('_')[0]
                            print(f">> TỪ CHỐI: Khuon mat da ton tai trong he thong voi ten '{matched_name}' (Độ khớp: {best_score:.2f})!")
                            cap.release()
                            cv2.destroyAllWindows()
                            return jsonify({"status": "error", "message": f"Khuôn mặt đã tồn tại dưới tên {matched_name}"}), 400

                    is_checked_existence = True 

                if frame_count % 10 == 0:
                    captured_count += 1
                    path = os.path.join(DB_PATH, f"{name}_{captured_count}.jpg")
                    cv2.imwrite(path, img)
                    print(f"  -> Da luu anh {captured_count}/{target_images} tai frame {frame_count}")

        cap.release()
        cv2.destroyAllWindows()

        if captured_count > 0:
            print(f">> DA CHUP XONG {captured_count} ANH CHO {name}. DANG NAP VAO BO NHO...")
            with db_lock:
                if os.path.exists(VECTOR_DB_PATH):
                    os.remove(VECTOR_DB_PATH) 
            load_faces()
            # TRẢ VỀ JSON THÀNH CÔNG TẠI ĐÂY
            return jsonify({"status": "success", "message": f"OK - Da luu {captured_count} anh"}), 200
        else:
            print(f"\n>> LOI: Khong tim thay khuon mat nao sau {frame_count} frame!")
            return jsonify({"status": "error", "message": "Khong tim thay khuon mat"}), 400

    except Exception as e:
        print("=== LOI KHI DANG KY ===")
        print(str(e))
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        # TRẢ VỀ LỖI TẠI ĐÂY (TRƯỚC ĐÓ BẠN ĐỂ THÔNG BÁO THÀNH CÔNG Ở ĐÂY)
        return jsonify({"status": "error", "message": "Loi he thong"}), 500

# ================= LIỆT KÊ TÊN NGƯỜI DÙNG =================
@app.route('/list', methods=['GET'])
@token_required
def list_users():
    unique_names = list(set([n.split('_')[0] for n in known_names]))
    return jsonify(unique_names)

# ================= XÓA KHUÔN MẶT THEO TÊN =================
@app.route('/delete', methods=['POST', 'OPTIONS'])
@token_required
def delete_user():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        name = request.json.get('name') or request.args.get('name')
        
        if not name:
            return jsonify({"status": "error", "message": "Thiếu tham số name"}), 400
        
        print(f"\n--- BAT DAU XOA KHUON MAT: {name} ---")
        
        deleted_count = 0
        with db_lock:
            # Xóa tất cả ảnh của người dùng
            for file in os.listdir(DB_PATH):
                if file.startswith(f"{name}_") and file.endswith(('.jpg', '.png')):
                    file_path = os.path.join(DB_PATH, file)
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"  -> Xóa: {file}")
        
        if deleted_count == 0:
            return jsonify({"status": "error", "message": f"Không tìm thấy khuôn mặt của {name}"}), 404
        
        # Rebuild database vector
        print(f">> Đã xóa {deleted_count} ảnh của {name}. Đang rebuild database...")
        with db_lock:
            if os.path.exists(VECTOR_DB_PATH):
                os.remove(VECTOR_DB_PATH)
        load_faces()
        
        print(f">> Xóa thành công!")
        return jsonify({"status": "success", "message": f"Đã xóa {deleted_count} ảnh của {name}"}), 200
        
    except Exception as e:
        print(f"=== LỖI KHI XÓA ===")
        print(str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Lỗi hệ thống"}), 500

# ================= KHỞI CHẠY SERVER =================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)