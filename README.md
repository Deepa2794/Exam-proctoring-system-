import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLO model
def load_yolo_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model, device

# Run YOLO inference
def detect_objects(model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize for YOLO
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)  # Normalize and prepare batch
    img = img.to(device)

    with torch.no_grad():
        preds = model(img)
        detections = non_max_suppression(preds, conf_thres=0.4, iou_thres=0.5)

    return detections

# Process detections and flag suspicious activities
def process_detections(detections, frame, width, height):
    face_count = 0
    suspicious = False
    if detections[0] is not None:
        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = scale_coords((640, 640), (x1, y1, x2, y2), (width, height)).round()

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}" if int(cls) == 0 else f"Object {int(cls)}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Count faces and detect suspicious behavior
            if int(cls) == 0:  # Class 0 is "person/face"
                face_count += 1
                if face_count > 1:  # More than one face
                    suspicious = True

    return frame, face_count, suspicious

def main():
    # Load YOLO model
    weights_path = 'yolov5/weights/yolov5s.pt'  # Adjust the path
    model, device = load_yolo_model(weights=weights_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam")
            break

        height, width = frame.shape[:2]

        # Detect objects using YOLO
        detections = detect_objects(model, device, frame)

        # Process detections for monitoring
        processed_frame, face_count, suspicious = process_detections(detections, frame, width, height)

        # Display warning for suspicious activities
        if face_count == 0:
            cv2.putText(processed_frame, "Warning: Candidate not detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif suspicious:
            cv2.putText(processed_frame, "Warning: Multiple faces detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("AI-Based Proctoring", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
HTML
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Proctoring Dashboard</title>
</head>
<body>
    <h1>Welcome to the AI-Based Proctoring Dashboard</h1>
    <nav>
        <ul>
            <li><a href="/attendance">View Attendance</a></li>
            <li><a href="/malpractice">View Malpractice Reports</a></li>
        </ul>
    </nav>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Attendance Report</title>
</head>
<body>
    <h1>Attendance Report</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Attendance Status</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record.student_name }}</td>
            <td>{{ record.exam_name }}</td>
            <td>{{ record.attendance_status }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Malpractice Reports</title>
</head>
<body>
    <h1>Malpractice Reports</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Issue Detected</th>
        </tr>
        {% for report in reports %}
        <tr>
            <td>{{ report.student_name }}</td>
            <td>{{ report.exam_name }}</td>
            <td>{{ report.issue_detected }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLO model
def load_yolo_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model, device

# Run YOLO inference
def detect_objects(model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize for YOLO
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)  # Normalize and prepare batch
    img = img.to(device)

    with torch.no_grad():
        preds = model(img)
        detections = non_max_suppression(preds, conf_thres=0.4, iou_thres=0.5)

    return detections

# Process detections and flag suspicious activities
def process_detections(detections, frame, width, height):
    face_count = 0
    suspicious = False
    if detections[0] is not None:
        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = scale_coords((640, 640), (x1, y1, x2, y2), (width, height)).round()

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}" if int(cls) == 0 else f"Object {int(cls)}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Count faces and detect suspicious behavior
            if int(cls) == 0:  # Class 0 is "person/face"
                face_count += 1
                if face_count > 1:  # More than one face
                    suspicious = True

    return frame, face_count, suspicious

def main():
    # Load YOLO model
    weights_path = 'yolov5/weights/yolov5s.pt'  # Adjust the path
    model, device = load_yolo_model(weights=weights_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam")
            break

        height, width = frame.shape[:2]

        # Detect objects using YOLO
        detections = detect_objects(model, device, frame)

        # Process detections for monitoring
        processed_frame, face_count, suspicious = process_detections(detections, frame, width, height)

        # Display warning for suspicious activities
        if face_count == 0:
            cv2.putText(processed_frame, "Warning: Candidate not detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif suspicious:
            cv2.putText(processed_frame, "Warning: Multiple faces detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("AI-Based Proctoring", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
HTML
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Proctoring Dashboard</title>
</head>
<body>
    <h1>Welcome to the AI-Based Proctoring Dashboard</h1>
    <nav>
        <ul>
            <li><a href="/attendance">View Attendance</a></li>
            <li><a href="/malpractice">View Malpractice Reports</a></li>
        </ul>
    </nav>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Attendance Report</title>
</head>
<body>
    <h1>Attendance Report</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Attendance Status</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record.student_name }}</td>
            <td>{{ record.exam_name }}</td>
            <td>{{ record.attendance_status }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Malpractice Reports</title>
</head>
<body>
    <h1>Malpractice Reports</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Issue Detected</th>
        </tr>
        {% for report in reports %}
        <tr>
            <td>{{ report.student_name }}</td>
            <td>{{ report.exam_name }}</td>
            <td>{{ report.issue_detected }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLO model
def load_yolo_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model, device

# Run YOLO inference
def detect_objects(model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize for YOLO
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)  # Normalize and prepare batch
    img = img.to(device)

    with torch.no_grad():
        preds = model(img)
        detections = non_max_suppression(preds, conf_thres=0.4, iou_thres=0.5)

    return detections

# Process detections and flag suspicious activities
def process_detections(detections, frame, width, height):
    face_count = 0
    suspicious = False
    if detections[0] is not None:
        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = scale_coords((640, 640), (x1, y1, x2, y2), (width, height)).round()

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}" if int(cls) == 0 else f"Object {int(cls)}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Count faces and detect suspicious behavior
            if int(cls) == 0:  # Class 0 is "person/face"
                face_count += 1
                if face_count > 1:  # More than one face
                    suspicious = True

    return frame, face_count, suspicious

def main():
    # Load YOLO model
    weights_path = 'yolov5/weights/yolov5s.pt'  # Adjust the path
    model, device = load_yolo_model(weights=weights_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam")
            break

        height, width = frame.shape[:2]

        # Detect objects using YOLO
        detections = detect_objects(model, device, frame)

        # Process detections for monitoring
        processed_frame, face_count, suspicious = process_detections(detections, frame, width, height)

        # Display warning for suspicious activities
        if face_count == 0:
            cv2.putText(processed_frame, "Warning: Candidate not detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif suspicious:
            cv2.putText(processed_frame, "Warning: Multiple faces detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("AI-Based Proctoring", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
HTML
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Proctoring Dashboard</title>
</head>
<body>
    <h1>Welcome to the AI-Based Proctoring Dashboard</h1>
    <nav>
        <ul>
            <li><a href="/attendance">View Attendance</a></li>
            <li><a href="/malpractice">View Malpractice Reports</a></li>
        </ul>
    </nav>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Attendance Report</title>
</head>
<body>
    <h1>Attendance Report</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Attendance Status</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record.student_name }}</td>
            <td>{{ record.exam_name }}</td>
            <td>{{ record.attendance_status }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Malpractice Reports</title>
</head>
<body>
    <h1>Malpractice Reports</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Issue Detected</th>
        </tr>
        {% for report in reports %}
        <tr>
            <td>{{ report.student_name }}</td>
            <td>{{ report.exam_name }}</td>
            <td>{{ report.issue_detected }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLO model
def load_yolo_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model, device

# Run YOLO inference
def detect_objects(model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize for YOLO
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)  # Normalize and prepare batch
    img = img.to(device)

    with torch.no_grad():
        preds = model(img)
        detections = non_max_suppression(preds, conf_thres=0.4, iou_thres=0.5)

    return detections

# Process detections and flag suspicious activities
def process_detections(detections, frame, width, height):
    face_count = 0
    suspicious = False
    if detections[0] is not None:
        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = scale_coords((640, 640), (x1, y1, x2, y2), (width, height)).round()

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}" if int(cls) == 0 else f"Object {int(cls)}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Count faces and detect suspicious behavior
            if int(cls) == 0:  # Class 0 is "person/face"
                face_count += 1
                if face_count > 1:  # More than one face
                    suspicious = True

    return frame, face_count, suspicious

def main():
    # Load YOLO model
    weights_path = 'yolov5/weights/yolov5s.pt'  # Adjust the path
    model, device = load_yolo_model(weights=weights_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam")
            break

        height, width = frame.shape[:2]

        # Detect objects using YOLO
        detections = detect_objects(model, device, frame)

        # Process detections for monitoring
        processed_frame, face_count, suspicious = process_detections(detections, frame, width, height)

        # Display warning for suspicious activities
        if face_count == 0:
            cv2.putText(processed_frame, "Warning: Candidate not detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif suspicious:
            cv2.putText(processed_frame, "Warning: Multiple faces detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("AI-Based Proctoring", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
HTML
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    attendance_status = db.Column(db.String(50), nullable=False)  # Present/Absent

class MalpracticeReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)
    issue_detected = db.Column(db.String(200), nullable=False)  # Description of malpractice

# Initialize database (Run this only once)
@app.before_first_request
def create_tables():
    db.create_all()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Attendance page
@app.route('/attendance')
def attendance():
    records = Attendance.query.all()
    return render_template('attendance.html', records=records)

# Malpractice report page
@app.route('/malpractice')
def malpractice():
    reports = MalpracticeReport.query.all()
    return render_template('malpractice.html', reports=reports)

# Add data to attendance (for demonstration)
@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    attendance_status = request.form['attendance_status']
    new_record = Attendance(student_name=student_name, exam_name=exam_name, attendance_status=attendance_status)
    db.session.add(new_record)
    db.session.commit()
    return "Attendance added!"

# Add data to malpractice report (for demonstration)
@app.route('/add_malpractice', methods=['POST'])
def add_malpractice():
    student_name = request.form['student_name']
    exam_name = request.form['exam_name']
    issue_detected = request.form['issue_detected']
    new_report = MalpracticeReport(student_name=student_name, exam_name=exam_name, issue_detected=issue_detected)
    db.session.add(new_report)
    db.session.commit()
    return "Malpractice report added!"

if __name__ == "__main__":
    app.run(debug=True)
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Proctoring Dashboard</title>
</head>
<body>
    <h1>Welcome to the AI-Based Proctoring Dashboard</h1>
    <nav>
        <ul>
            <li><a href="/attendance">View Attendance</a></li>
            <li><a href="/malpractice">View Malpractice Reports</a></li>
        </ul>
    </nav>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Attendance Report</title>
</head>
<body>
    <h1>Attendance Report</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Attendance Status</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record.student_name }}</td>
            <td>{{ record.exam_name }}</td>
            <td>{{ record.attendance_status }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Malpractice Reports</title>
</head>
<body>
    <h1>Malpractice Reports</h1>
    <table border="1">
        <tr>
            <th>Student Name</th>
            <th>Exam Name</th>
            <th>Issue Detected</th>
        </tr>
        {% for report in reports %}
        <tr>
            <td>{{ report.student_name }}</td>
            <td>{{ report.exam_name }}</td>
            <td>{{ report.issue_detected }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>yolov5.models.commonyolov5.models.common
