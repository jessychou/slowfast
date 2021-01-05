import face_recognition
import cv2
import numpy as np
import time
from tqdm import tqdm
import os

from src.settings import FR_DATA_FOLDER, VIDEO_FOLDER


def _get_face_encodings():
    ken_image = face_recognition.load_image_file(os.path.join(FR_DATA_FOLDER, 'ken_test.jpg'))
    ken_face_encoding = face_recognition.face_encodings(ken_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(os.path.join(FR_DATA_FOLDER, 'biden.jpg'))
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        ken_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "Ken Wong",
        "Joe Biden"
    ]
    return known_face_encodings, known_face_names


def recognize_faces(input_filename):
    # get face encodings
    known_face_encodings, known_face_names = _get_face_encodings()
    input_video_capture = cv2.VideoCapture(input_filename)

    # Define the codec and create VideoWriter object
    fps = input_video_capture.get(cv2.CAP_PROP_FPS)
    width = int(input_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(input_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out_video_filename = os.path.join(VIDEO_FOLDER, 'fr_output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_filename, fourcc, fps, (width,  height))

    print("Running face recognition")
    start = time.time()
    for i in tqdm(range(total_frames)):
        # Grab a single frame of video
        ret, frame = input_video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = frame[:, :, ::-1]

        # Only process every fourth frame of video to save time
        # if i % 100 == 0:
        if True:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                #
                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        # cv2.imshow('Video', frame)

        # write the flipped frame
        out.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    print(f'Face recognition finished. Total time is {end-start}')
    # Release handle to the webcam
    input_video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    return out_video_filename
