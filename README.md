# Real-Time Face Recognition with Facial Landmarks and Database Storage

## Overview

This project implements a real-time face recognition system using Python. It detects faces in a video stream, computes face embeddings using the [face_recognition](https://github.com/ageitgey/face_recognition) library, and uses a SQLite database to store and match these embeddings. When a face is detected, the system either recognizes it as an existing person or adds it as a new person (named "person1", "person2", etc.). Additionally, the program draws both a bounding box and a facial landmark "mask" on each detected face.

## Features

- **Real-Time Face Detection:**  
  Captures video from your built-in camera and processes frames in real time.
- **Face Embedding and Matching:**  
  Uses face embeddings to uniquely identify faces. Matches are stored in and compared with a SQLite database.
- **Facial Landmark Masking:**  
  Draws detailed facial landmarks (mask) on each detected face.
- **Performance Optimization:**  
  Processes a downscaled version of the video frame for faster detection, then scales the coordinates back for display.

