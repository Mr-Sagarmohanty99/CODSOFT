import cv2
import face_recognition
import numpy as np

# Load a sample picture and learn how to recognize it.
known_face_encodings = []
known_face_names = []

# Load known faces and their names
def load_known_faces():
    # Example: Load an image and encode it
    image = face_recognition.load_image_file("known_faces/person1.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("Person 1")

    # Add more known faces as needed
    # Repeat for each known person
    # image = face_recognition.load_image_file("known_faces/person2.jpg")
    # encoding = face_recognition.face_encodings(image)[0]
    # known_face_encodings.append(encoding)
    # known_face_names.append("Person 2")

load_known_faces()

# Face detection and recognition function for an image
def recognize_faces_in_image(image_path):
    # Load the image and convert from BGR to RGB
    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces and face encodings in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Loop over each face in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the closest known face if any matches
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face and add the name
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Face detection and recognition function for video
def recognize_faces_in_video(video_path=0):
    # Open video file or capture device
    video_capture = cv2.VideoCapture(video_path)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop over each face in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encoding with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the closest known face if any matches
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face and add the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage for image recognition
# recognize_faces_in_image("test_image.jpg")

# Example usage for video recognition (0 is the webcam)
recognize_faces_in_video(0)