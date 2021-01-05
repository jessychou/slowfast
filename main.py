from src.face_recognition_component import recognize_faces
from src.settings import VIDEO_FOLDER
import os

if __name__ == '__main__':

    input_fname = os.path.join(VIDEO_FOLDER, "video.mp4")
    # face recognition
    output_fname = recognize_faces(input_filename=input_fname)
