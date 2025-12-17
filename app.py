import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import os
import base64
import time
from collections import defaultdict, deque
import threading
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Hazard Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with improved contrast and modern design
st.markdown("""
<style>
    .main-header {
       
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Enhanced Alert Boxes with Better Contrast */
    .alert-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .alert-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 5px;
        border-radius: 0 12px 12px 0;
    }
    
    .alert-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* High Risk - Red Alert */
    .danger { 
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .danger::before { background: #ffffff; }
    
    /* Medium Risk - Orange/Yellow Alert */
    .warning { 
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .warning::before { background: #ffffff; }
    
    /* Safe/Good Status - Green */
    .safe { 
        background: linear-gradient(135deg, #26d0ce 0%, #1dd1a1 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .safe::before { background: #ffffff; }
    
    /* Info/General - Blue */
    .info { 
        background: linear-gradient(135deg, #54a0ff 0%, #2e86de 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .info::before { background: #ffffff; }
    
    /* Lane Departure Warning - Purple */
    .lane-warning {
        background: linear-gradient(135deg, #a55eea 0%, #8b5cf6 100%);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .lane-warning::before { background: #ffffff; }
    
    /* Alert Icons */
    .alert-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Enhanced Metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Button Improvements */
    .stButton > button {
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Alert Animation */
    @keyframes alertPulse {
        0% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.8; transform: scale(1); }
    }
    
    .alert-high-priority {
        animation: alertPulse 2s infinite;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-danger { background-color: #ff4757; }
    .status-warning { background-color: #ffa726; }
    .status-safe { background-color: #1dd1a1; }
    .status-info { background-color: #54a0ff; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Enhanced Footer */
    .footer-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
</style>
            
""", unsafe_allow_html=True)

import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your background image path
add_bg_from_local(r"C:\Users\BIT\OneDrive\Desktop\behaviour\tata.jpg")




class AdvancedHazardDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.alert_cooldown = {}
        self.tracking_history = defaultdict(deque)
        self.detection_counts = defaultdict(int)
        self.speed_estimates = {}
        self.danger_zones = []
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Advanced detection parameters
        self.collision_threshold = 0.3
        self.speed_threshold = 50  # pixels per frame
        self.tracking_frames = 10
        
        # Hazard categories with risk levels
        self.hazard_categories = {
            
            'bicycle': {'risk': 'HIGH', 'color': (0, 165, 255), 'sound': 'Cyclist ahead!'},
            'motorcycle': {'risk': 'HIGH', 'color': (255, 0, 255), 'sound': 'Motorcycle detected!'},
            'car': {'risk': 'MEDIUM', 'color': (0, 255, 255), 'sound': 'Vehicle approaching!'},
            'bus': {'risk': 'MEDIUM', 'color': (255, 255, 0), 'sound': 'Large vehicle detected!'},
            'truck': {'risk': 'MEDIUM', 'color': (255, 165, 0), 'sound': 'Truck detected!'},
            'traffic light': {'risk': 'INFO', 'color': (0, 255, 0), 'sound': 'Traffic light ahead!'},
            'stop sign': {'risk': 'INFO', 'color': (255, 0, 0), 'sound': 'Stop sign detected!'}
        }
    
    def calculate_collision_risk(self, box1, box2):
        """Calculate collision risk between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        # IoU (Intersection over Union)
        iou = intersection / union if union > 0 else 0
        
        # Distance between centers
        center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
        distance = euclidean(center1, center2)
        
        # Normalize distance by frame size
        normalized_distance = distance / 1000  # Assuming max distance ~1000 pixels
        
        # Risk score (higher is riskier)
        risk_score = (iou * 0.7) + ((1 - normalized_distance) * 0.3)
        return risk_score
    
    def estimate_speed(self, obj_id, current_pos):
        """Estimate object speed based on position history"""
        if obj_id in self.tracking_history:
            history = self.tracking_history[obj_id]
            if len(history) >= 2:
                prev_pos = history[-1]
                speed = euclidean(current_pos, prev_pos)
                self.speed_estimates[obj_id] = speed
                return speed
        
        self.tracking_history[obj_id].append(current_pos)
        if len(self.tracking_history[obj_id]) > self.tracking_frames:
            self.tracking_history[obj_id].popleft()
        
        return 0
    
    def detect_lane_departure(self, frame, vehicle_boxes):
        """Advanced lane detection with departure warnings"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        height, width = frame.shape[:2]
        
        # Define region of interest (trapezoid)
        roi_vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.6), int(height * 0.6)),
            (int(width * 0.4), int(height * 0.6))
        ]], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Line Transform with optimized parameters
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=100, 
            maxLineGap=50
        )
        
        left_lines, right_lines = [], []
        lane_departure_warning = ""
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                
                # Classify lines as left or right lane markers
                if slope < -0.3:  # Left lane (negative slope)
                    left_lines.append(line[0])
                elif slope > 0.3:  # Right lane (positive slope)
                    right_lines.append(line[0])
            
            # Draw lane lines
            for line in left_lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            for line in right_lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # Check for lane departure (simplified check)
            if len(left_lines) == 0:
                lane_departure_warning = "⚠️ Left Lane Departure Risk!"
            elif len(right_lines) == 0:
                lane_departure_warning = "⚠️ Right Lane Departure Risk!"
        
        return frame, lane_departure_warning
    
    def analyze_frame(self, frame):
        """Main frame analysis function"""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if self.frame_count % 30 == 0:
            self.fps = 30 / (current_time - self.start_time)
            self.start_time = current_time
        
        # Run YOLO detection
        results = self.model.predict(
            source=frame, 
            conf=0.3, 
            save=False, 
            show=False, 
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        height, width = frame.shape[:2]
        
        detected_objects = []
        active_alerts = []
        
        # Process each detection
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            label = self.model.model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Object properties
            obj_width = x2 - x1
            obj_height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Estimate speed
            speed = self.estimate_speed(f"{label}_{i}", (center_x, center_y))
            
            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'speed': speed,
                'size_ratio': (obj_width * obj_height) / (width * height)
            })
            
            # Check if object is in hazard categories
            if label in self.hazard_categories:
                hazard_info = self.hazard_categories[label]
                risk_level = hazard_info['risk']
                
                # Define danger zones based on object type and position
                danger_zone = False
                alert_message = ""
                
                # Pedestrian-specific logic
                if label == 'person':
                    if center_y > height * 0.7:  # Lower part of frame
                        danger_zone = True
                        alert_message = "Slow down"
                    elif speed > 20:  # Fast-moving person
                        alert_message = "Slow down"
                
                # Vehicle-specific logic
                elif label in ['car', 'bus', 'truck']:
                    size_threshold = 0.3 if label in ['bus', 'truck'] else 0.25
                    if obj_height > height * size_threshold:
                        danger_zone = True
                        alert_message = f"🚗 {label.title()} too close!"
                    elif speed > self.speed_threshold:
                        alert_message = f"💨 Fast {label} approaching!"
                
                # Cyclist/Motorcyclist logic
                elif label in ['bicycle', 'motorcycle']:
                    if center_y > height * 0.6:
                        danger_zone = True
                        alert_message = f"🚴‍♂️ {label.title()} ahead!"
                
                # Traffic signs logic
                elif label in ['traffic light', 'stop sign']:
                    alert_message = f"🚦 {label.replace('_', ' ').title()} detected"
                
                if alert_message:
                    active_alerts.append({
                        'message': alert_message,
                        'risk': risk_level,
                        'position': (center_x, center_y),
                        'sound': hazard_info['sound']
                    })
                    
                    # Draw enhanced bounding box
                    color = hazard_info['color']
                    thickness = 4 if danger_zone else 2
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add risk level indicator
                    risk_color = (0, 0, 255) if risk_level == 'HIGH' else (0, 165, 255) if risk_level == 'MEDIUM' else (0, 255, 0)
                    cv2.putText(annotated_frame, f"{risk_level}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        
        # Lane detection
        annotated_frame, lane_warning = self.detect_lane_departure(annotated_frame, detected_objects)
        if lane_warning:
            active_alerts.append({
                'message': lane_warning,
                'risk': 'WARNING',
                'position': (width//2, 50),
                'sound': 'Lane departure warning!'
            })
        
        # Add system information overlay
        info_text = [
            f"FPS: {self.fps:.1f}",
            f"Objects: {len(detected_objects)}",
            f"Frame: {self.frame_count}",
            f"Alerts: {len(active_alerts)}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Add alert messages on frame
        for i, alert in enumerate(active_alerts):
            y_pos = height - 100 + (i * 30)
            color = (0, 0, 255) if alert['risk'] == 'HIGH' else (0, 165, 255) if alert['risk'] == 'MEDIUM' else (0, 255, 0)
            cv2.putText(annotated_frame, alert['message'], (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return annotated_frame, active_alerts, detected_objects



def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Advanced Smart Hazard Detection System</h1>
        <p>AI-Powered Real-time Traffic Safety Analysis with Audio Alerts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = AdvancedHazardDetector()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
        enable_audio = st.checkbox("Enable Audio Alerts", value=True)
        enable_lane_detection = st.checkbox("Enable Lane Detection", value=True)
        
        # Risk level filters
        st.subheader("Alert Filters")
        show_high_risk = st.checkbox("High Risk Alerts", value=True)
        show_medium_risk = st.checkbox("Medium Risk Alerts", value=True)
        show_info_alerts = st.checkbox("Info Alerts", value=False)
        
        # Enhanced System Status in Sidebar
        st.markdown("""
        <div style="background: linear-gradient(135deg, #54a0ff 0%, #2e86de 100%); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0; color: white;">
            <h4 style="margin: 0; text-align: center;">📊 System Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'current_fps'):
            fps_value = st.session_state.get('current_fps', 0)
            fps_color = "#1dd1a1" if fps_value > 20 else "#ffa726" if fps_value > 10 else "#ff4757"
            
            st.markdown(f'''
            <div style="background: {fps_color}; padding: 0.8rem; border-radius: 8px; 
                        color: white; text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 1.5rem; font-weight: bold;">{fps_value:.1f}</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">FPS</div>
            </div>
            ''', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'total_detections'):
            detection_count = st.session_state.get('total_detections', 0)
            
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #a55eea 0%, #8b5cf6 100%); 
                        padding: 0.8rem; border-radius: 8px; color: white; 
                        text-align: center; margin: 0.5rem 0;">
                <div style="font-size: 1.5rem; font-weight: bold;">{detection_count}</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Detections</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Driving Video", 
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a video file to analyze for hazards and lane detection"
        )
        
        if uploaded_file is not None:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_filename = tfile.name
            
            # Video processing
            cap = cv2.VideoCapture(temp_filename)
            
            if not cap.isOpened():
                st.error("Error: Could not open video file")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.info(f"📹 Video loaded: {total_frames} frames at {fps:.1f} FPS")
            
            # Control buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                start_analysis = st.button("🎬 Start Analysis", type="primary")
            with col_btn2:
                stop_analysis = st.button("⏹️ Stop Analysis")
            with col_btn3:
                reset_system = st.button("🔄 Reset System")
            
            if reset_system:
                st.session_state.detector = AdvancedHazardDetector()
                st.rerun()
            
            # Video display placeholder
            video_placeholder = st.empty()
            
            # Progress tracking
            progress_bar = st.progress(0)
            frame_info = st.empty()
            
            if start_analysis:
                # Initialize tracking variables
                st.session_state.analysis_data = {
                    'alerts': [],
                    'detections': [],
                    'frame_count': 0
                }
                
                frame_count = 0
                detector = st.session_state.detector
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    # Process frame
                    processed_frame, alerts, detections = detector.analyze_frame(frame)
                    
                    # Store data for analysis
                    st.session_state.analysis_data['alerts'].extend(alerts)
                    st.session_state.analysis_data['detections'].extend(detections)
                    st.session_state.analysis_data['frame_count'] = frame_count
                    
                    # Update session state for sidebar
                    st.session_state.current_fps = detector.fps
                    st.session_state.total_detections = len(detections)
                    
                    # Resize frame for display
                    display_frame = cv2.resize(processed_frame, (800, 600))
                    video_placeholder.image(display_frame, channels="BGR", use_column_width=True)
                    
                    # Update frame info
                    frame_info.text(f"Frame {frame_count}/{total_frames} | FPS: {detector.fps:.1f}")
                    
                    # Process alerts
                    for alert in alerts:
                        if enable_audio:
                            # Filter alerts based on risk level
                            should_play = (
                                (alert['risk'] == 'HIGH' and show_high_risk) or
                                (alert['risk'] == 'MEDIUM' and show_medium_risk) or
                                (alert['risk'] in ['INFO', 'WARNING'] and show_info_alerts)
                            )
                            
                            
                    
                    # Small delay to control playback speed
                    time.sleep(0.03)  # ~30 FPS display
            
            # Cleanup
            cap.release()
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    with col2:
        # Enhanced Real-time Alerts Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 12px; margin-bottom: 1rem; color: white;">
            <h3 style="margin: 0; text-align: center;">🚨 Real-time Alerts</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Current alerts display with enhanced styling
        if hasattr(st.session_state, 'analysis_data') and st.session_state.analysis_data['alerts']:
            recent_alerts = st.session_state.analysis_data['alerts'][-8:]  # Last 8 alerts for better display
            
            if recent_alerts:
                for i, alert in enumerate(reversed(recent_alerts)):  # Show most recent first
                    risk_class = alert['risk'].lower()
                    
                    # Determine alert styling and icon
                    if risk_class == 'high':
                        alert_class = "danger alert-high-priority"
                        icon = "🔴"
                        status_class = "status-danger"
                    elif risk_class == 'medium':
                        alert_class = "warning"
                        icon = "🟡"
                        status_class = "status-warning"
                    elif risk_class == 'warning':
                        alert_class = "lane-warning"
                        icon = "🟣"
                        status_class = "status-warning"
                    else:
                        alert_class = "info"
                        icon = "🔵"
                        status_class = "status-info"
                    
                    # Enhanced alert display with icons and status indicators
                    alert_html = f'''
                    <div class="alert-box {alert_class}">
                        <span class="alert-icon">{icon}</span>
                        <span class="status-indicator {status_class}"></span>
                        {alert["message"]}
                        <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">
                            Risk Level: {alert["risk"]}
                        </div>
                    </div>
                    '''
                    st.markdown(alert_html, unsafe_allow_html=True)
            else:
                # No alerts - show safe status
                st.markdown('''
                <div class="alert-box safe">
                    <span class="alert-icon">✅</span>
                    <span class="status-indicator status-safe"></span>
                    All Clear - No Active Hazards
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">
                        System Status: Monitoring
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            # System starting up
            st.markdown('''
            <div class="alert-box info">
                <span class="alert-icon">🔄</span>
                <span class="status-indicator status-info"></span>
                System Initializing...
                <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;">
                    Waiting for video input
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Enhanced Detection Statistics Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1dd1a1 0%, #26d0ce 100%); 
                    padding: 1rem; border-radius: 12px; margin: 1.5rem 0; color: white;">
            <h3 style="margin: 0; text-align: center;">📈 Detection Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'analysis_data') and st.session_state.analysis_data['detections']:
            detections_df = pd.DataFrame(st.session_state.analysis_data['detections'])
            
            # Object count chart with enhanced styling
            if not detections_df.empty:
                obj_counts = detections_df['label'].value_counts()
                
                # Enhanced bar chart
                fig = px.bar(
                    x=obj_counts.index, 
                    y=obj_counts.values,
                    title="🎯 Object Detection Count",
                    labels={'x': 'Object Type', 'y': 'Detection Count'},
                    color=obj_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font_size=16,
                    showlegend=False
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level distribution with enhanced styling
                alerts_df = pd.DataFrame(st.session_state.analysis_data['alerts'])
                if not alerts_df.empty:
                    risk_counts = alerts_df['risk'].value_counts()
                    
                    # Custom colors for risk levels
                    risk_colors = {
                        'HIGH': '#ff4757',
                        'MEDIUM': '#ffa726', 
                        'WARNING': '#a55eea',
                        'INFO': '#54a0ff'
                    }
                    colors = [risk_colors.get(risk, '#54a0ff') for risk in risk_counts.index]
                    
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="⚠️ Alert Risk Distribution",
                        color_discrete_sequence=colors
                    )
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font_size=16
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Quick Stats Summary
                total_detections = len(detections_df)
                total_alerts = len(alerts_df) if not alerts_df.empty else 0
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.markdown(f'''
                    <div class="metric-container">
                        <div style="font-size: 2rem; font-weight: bold;">{total_detections}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Total Objects</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_stat2:
                    st.markdown(f'''
                    <div class="metric-container">
                        <div style="font-size: 2rem; font-weight: bold;">{total_alerts}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Total Alerts</div>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            # No data available - show placeholder
            st.markdown('''
            <div style="text-align: center; padding: 2rem; color: #666; 
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        border-radius: 12px; margin: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h4>No Detection Data Available</h4>
                <p>Upload and analyze a video to see detection statistics</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-container">
        <h3 style="margin-bottom: 1rem;">🚗 Advanced Hazard Detection System v2.0</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Powered by YOLOv8 & Computer Vision</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">⚠️ For testing purposes only. Always maintain situational awareness while driving.</p>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
            Built using Streamlit, OpenCV, and Ultralytics YOLO
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()