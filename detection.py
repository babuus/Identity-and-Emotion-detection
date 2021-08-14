import face_recognition
import cv2
import numpy as np
from tensorflow import keras
import time
import os

def cam_data(sec):
    video_capture = cv2.VideoCapture(0)

    #imgs - image path
    #face_name - face name
    #face - face encoding
    imgs, known_face_encodings, known_face_names  = [], [], []

    #image path
    path = "C:/Users/usbab/Desktop/AI CWEM/cwem/Images"
    valid_images = [".jpg",".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        #imgs path
        imgs.append(os.path.join(path,f))
        print(f)
        
        #save names
        known_face_names.append(f.split('.')[0])
        
        # face image load
        face_image = face_recognition.load_image_file(os.path.join(path,f))
        face = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face)

    emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    model = keras.models.load_model('C:/Users/usbab/Desktop/AI CWEM/cwem/Model/Model_emotions.h5')
    start_time = time.time()
    
    prediction_arr = []
    name_arr = []
    while True:
        pred =""
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            if name in known_face_names:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(gray, (48,48)) 
                face = face/255.0
                predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
                pred = emotions[predictions]
                prediction_arr.append(pred)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name+" "+pred, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            name_arr.append(name)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time >= sec: #<---- Check if "" sec passed
            print(" 5sec over!")
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    return name_arr, prediction_arr
# print(cam_data())