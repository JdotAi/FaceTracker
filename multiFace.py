import cv2
import face_recognition
import sqlite3
import json
import numpy as np

# ----------------------------------
# Database Setup: SQLite
# ----------------------------------
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding TEXT
)
''')
conn.commit()

# ----------------------------------
# Utility Functions
# ----------------------------------
def get_matching_face(embedding, cursor, threshold=0.6):
    """
    Check the database for a stored face embedding that is similar to the given embedding.
    Returns the associated name if a match is found; otherwise, returns None.
    """
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    for row in rows:
        stored_name = row[0]
        stored_embedding = np.array(json.loads(row[1]))
        distance = np.linalg.norm(embedding - stored_embedding)
        if distance < threshold:
            return stored_name
    return None

def add_new_face(embedding, cursor, conn):
    """
    Assign a new name (e.g., "person1", "person2", etc.) based on the current count in the database,
    add the new face embedding to the database, and return the new name.
    """
    cursor.execute("SELECT COUNT(*) FROM faces")
    count = cursor.fetchone()[0]
    new_name = f"person{count+1}"
    embedding_json = json.dumps(embedding.tolist())
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (new_name, embedding_json))
    conn.commit()
    return new_name

# ----------------------------------
# Main Loop: Face Detection and Recognition
# ----------------------------------
# Open the built-in camera.
cap = cv2.VideoCapture(0)

# To speed up detection, we use a scaling factor to work on a smaller frame.
scale_factor = 0.25  # Process at 25% of the original resolution

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing.
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    # Convert from BGR to RGB.
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute embeddings on the smaller frame.
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

    for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
        top, right, bottom, left = face_location
        # Scale coordinates back to original frame size.
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        # Get the matching face name from the database.
        matched_name = get_matching_face(face_encoding, cursor, threshold=0.6)
        if matched_name is None:
            matched_name = add_new_face(face_encoding, cursor, conn)
            print(f"Added new face: {matched_name}")
        else:
            print(f"{matched_name} is on the screen.")

        # Draw a rectangle around the face.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, matched_name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the facial landmarks (mask) for this face if available.
        if i < len(face_landmarks_list):
            landmarks = face_landmarks_list[i]
            # Iterate over each facial feature.
            for feature, points in landmarks.items():
                # Convert points to a numpy array and scale them up.
                pts = np.array(points, dtype=np.float32)
                pts = (pts / scale_factor).astype(np.int32)
                # Draw polylines over the feature.
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
