# Face Detection and Attendance System

This project is a comprehensive system designed for real-time face detection and attendance marking. It consists of a FastAPI backend for processing detection algorithms and a React frontend for displaying results and managing the system interface.

---

## Project Overview

The Face Detection and Attendance System leverages computer vision and machine learning techniques to:

- Detect faces in a live video feed.
- Identify phones in the feed to prevent misuse.
- Analyze dark surroundings and suspicious textures to enhance detection accuracy.
- Provide real-time liveness detection to distinguish real faces from static images.

---

## Setup Instructions

Follow the steps below to set up and run the project:

### Backend Setup

1. **Navigate to the Backend Directory:**
   ```bash
   cd backend
   ```

2. **Install Python Dependencies:**
   Ensure you have Python installed. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server:**
   Run the backend server by executing:
   ```bash
   python main.py
   ```

   The server will start at `http://localhost:8000` by default.

### Frontend Setup

1. **Install Node.js Dependencies:**
   Navigate to the frontend directory (if separate) and install dependencies:
   ```bash
   npm install
   ```

2. **Start the Development Server:**
   Launch the frontend server with:
   ```bash
   npm run dev
   ```

   The frontend will typically be hosted at `http://localhost:3000`.

---

## Features

### Real-Time Detection
- **Face Detection:** Identifies faces in the live camera feed and marks attendance.
- **Phone Detection:** Detects mobile phones in the feed to prevent unauthorized usage.

### Security Enhancements
- **Dark Surroundings Detection:** Identifies attempts to bypass the system in poorly lit conditions.
- **Texture Analysis:** Analyzes textures to detect static images used to spoof the system.
- **Liveness Detection:** Utilizes challenge-response tasks and movement analysis to confirm the presence of a live person.
- **Depth Analysis (Optional):** Uses stereo or IR sensors for additional security.

### User Interface
- **Live Camera Feed:** Displays a real-time feed with detection annotations.
- **Results Dashboard:** Shows detection logs and attendance records for analysis.

---

## Usage Instructions

### Running the Project

1. Ensure both backend and frontend servers are running as per the setup instructions.
2. Open the frontend URL in your browser (default: `http://localhost:3000`).
3. Use the system interface to:
   - View the live camera feed.
   - Monitor detection results.
   - Manage attendance records.

---

## Requirements

### Backend
- **Python** (version 3.8 or higher)
- **FastAPI** framework
- **OpenCV** library for computer vision
- **Dependencies:** Listed in `requirements.txt`

### Frontend
- **Node.js** (version 14 or higher)
- **React** framework

---

## Contributing

Contributions to improve the project are welcome. Please fork the repository and create a pull request with your changes.

---
## want complete project ?
 contact me for the complete project

---

## Contact

For any queries or issues, please contact the development team via the "Contact Us" page in the application.

