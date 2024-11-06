# Cogniable assignment updated:

# Changes Made: Instead of using the Haar Classifier, Changed it to use Dlib's CNN Face Detector for face detections
# Note: Using this requires quite a bit of Computation Capacity compared to the earlier version.

#importing libraries
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import cv2
import numpy as np
from torchvision import transforms
from collections import defaultdict
import pandas as pd
import dlib
from deepface import DeepFace

# load model using MTCNN
def initialize_models_2():
    yolo_model = YOLO("yolov8n-seg.pt")  # Load your YOLOv8 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # face_detector = dlib.get_frontal_face_detector()  # Dlib's HOG + SVM face detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    print("Models initialized.")
    # return yolo_model, face_detector, face_recognition_model, device
    return yolo_model, cnn_face_detector, face_recognition_model, device

def get_face_encoding(face_image, model, device):
    if face_image is None or face_image.size == 0:
        print("Empty face image")
        return None

    # Resize face image to 160x160 (size expected by InceptionResnetV1)
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the numpy array to PIL image
        transforms.Resize((160, 160)),  # Resize to the size expected by the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    face_tensor = transform(face_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        face_encoding = model(face_tensor).detach().cpu().numpy()

    return face_encoding

def video_analysis(video_path,frame_interval):
    # Initialize models
    # yolo_model, face_detector, face_recognition_model, device = initialize_models_2()
    yolo_model, cnn_face_detector, face_recognition_model, device = initialize_models_2()


    # Dictionary to store face encodings and IDs
    face_encodings = {}
    next_face_id = 1

    id_counter = defaultdict(int)
    person_details = defaultdict(lambda: {"entry": float('inf'), "exit": float('-inf'), "name": "Unknown"})

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    print("Processing video frames...")

    output_video_path = f'output_video_with_boxes_{video_path}.mp4'

    # Initialize frame counter
    frame_number = 0

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'MJPG' for other formats
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            # Set the video frame position
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number * frame_interval)

            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("End of video")
                break

            frame_number += 1

            # Use YOLOv8 to detect bodies and segment them
            results = yolo_model.track(frame, persist=True)

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_labels = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_label in zip(boxes, track_ids, class_labels):
                    if class_label == 0:  # Filter to only track persons
                        if id_counter[track_id] == 0:
                            id_counter[track_id] = max(id_counter.values(), default=0) + 1

                        x1, y1, x2, y2 = box
                        person_region = frame[y1:y2, x1:x2]
                        person_id = f"ID{id_counter[track_id]}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(frame, f"Person_id = {person_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Use Dlib's HOG + SVM to detect faces within the person region
                        gray_person_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
                        # faces = face_detector(gray_person_region,1)  # Upsample once for better detection
                        
                        # Use Dlib's CNN Face Detector to detect faces within the person region
                        rgb_person_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
                        faces = cnn_face_detector(rgb_person_region, 1)

                        face_coords = [None, None, None, None]
                        face_detected = False

                        for face in faces:
                                fx1, fy1, fx2, fy2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
                                # Ensure coordinates are within bounds
                                fx1 = max(0, fx1)
                                fy1 = max(0, fy1)
                                fx2 = min(person_region.shape[1], fx2)
                                fy2 = min(person_region.shape[0], fy2)
                                if fx1 >= fx2 or fy1 >= fy2:
                                    print("Invalid face coordinates, skipping this face.")
                                    continue
                                
                                face_image = person_region[fy1:fy2, fx1:fx2]

                                # Improve face image quality
                                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                                # Perform age estimation (assuming you have the DeepFace setup as before)
                                analysis = DeepFace.analyze(img_path=face_image, actions=['age'], enforce_detection=False)

                                for face_data in analysis:
                                    if isinstance(face_data, dict):
                                        predicted_age = face_data['age']
                                        if predicted_age >= 18:
                                            label = "Therapist"
                                            box_color = (0, 255, 0)  # Green
                                        else:
                                            label = "Child"
                                            box_color = (255, 0, 0)  # Blue
                                    else:
                                        label = "Unknown"
                                        box_color = (128, 128, 128)  # Gray

                                #     face_coords = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
                                #     cv2.rectangle(frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), box_color, 2)
                                #     cv2.putText(frame, f"{label} ({int(predicted_age)} yrs)", (face_coords[0], face_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                                
                                # use this code if you do not want to use the age part of the code
                                # label = "Unknown"
                                # box_color = (0, 0, 255)
                                # predicted_age = 0

                                # face_coords = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
                                # face_detected = True

                                # Compute face encoding using InceptionResnetV1
                                face_encoding = get_face_encoding(face_image, face_recognition_model, device)
                                if face_encoding is None:
                                    continue

                                # Comparing the detected face with known faces
                                min_distance = float('inf')
                                matched_id = None

                                for face_id, stored_encoding in face_encodings.items():
                                    distance = np.linalg.norm(stored_encoding - face_encoding)
                                    if distance < min_distance:
                                        min_distance = distance
                                        matched_id = face_id

                                if min_distance < 0.6:  # Adjust threshold based on validation
                                    face_id = matched_id
                                else:
                                    face_id = next_face_id
                                    face_encodings[face_id] = face_encoding
                                    next_face_id += 1

                                face_coords = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
                                face_detected = True
                                cv2.rectangle(frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), box_color, 2)
                                cv2.putText(frame, f"{label} ({int(predicted_age)} yrs)", (face_coords[0], face_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Display or save the frame with bounding boxes
            cv2.imshow("Processed Frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(f"Processed frame {frame_number * frame_interval}")

    finally:
        # Release handle to the video file
        video_capture.release()
        cv2.destroyAllWindows()
        out.release()
        print("Processing Complete")

if name =='__main__':
    video_path = "ABA Therapy_ Daniel - Communication.mp4"
    # video_path = "test2.mp4"
    frame_skip = 50
    video_analysis(video_path,frame_skip)