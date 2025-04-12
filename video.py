import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from PIL import Image, ImageTk
import cv2
from pathlib import Path
import threading
import time

class EnhancedVideoPlayer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.setup_ui()
        self.initialize_state()
        
    def setup_ui(self):
        # Main container for video player
        self.container = ctk.CTkFrame(self.parent_frame)
        self.container.pack(fill="both", expand=True)
        
        # Video canvas with black background
        self.canvas = tk.Canvas(
            self.container,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control bar
        self.controls = ctk.CTkFrame(self.container)
        self.controls.pack(fill="x", padx=10, pady=(0, 10))
        
        # Timeline slider
        self.timeline = ctk.CTkSlider(
            self.controls,
            from_=0,
            to=100,
            command=self.seek_video
        )
        self.timeline.pack(fill="x", padx=10, pady=5)
        
        # Playback controls
        self.button_frame = ctk.CTkFrame(self.controls, fg_color="transparent")
        self.button_frame.pack(fill="x", padx=10)
        
        # Play/Pause button
        self.play_pause_btn = ctk.CTkButton(
            self.button_frame,
            text="⏸",
            width=40,
            command=self.toggle_playback
        )
        self.play_pause_btn.pack(side="left", padx=5)
        
        # Restart button
        self.restart_btn = ctk.CTkButton(
            self.button_frame,
            text="⟲",
            width=40,
            command=self.restart_video
        )
        self.restart_btn.pack(side="left", padx=5)
        
        # Speed control
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_menu = ctk.CTkOptionMenu(
            self.button_frame,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            variable=self.speed_var,
            command=self.change_speed,
            width=70
        )
        self.speed_menu.pack(side="left", padx=5)
        
        # Time display
        self.time_label = ctk.CTkLabel(
            self.button_frame,
            text="0:00 / 0:00",
            width=100
        )
        self.time_label.pack(side="right", padx=5)
        
        # Volume control
        self.volume_slider = ctk.CTkSlider(
            self.button_frame,
            from_=0,
            to=100,
            width=100,
            command=self.change_volume
        )
        self.volume_slider.pack(side="right", padx=5)
        self.volume_slider.set(100)
        
        # Volume icon
        self.volume_btn = ctk.CTkButton(
            self.button_frame,
            text="🔊",
            width=40,
            command=self.toggle_mute
        )
        self.volume_btn.pack(side="right", padx=5)
        
    def initialize_state(self):
        self.cap = None
        self.is_playing = True
        self.playback_speed = 1.0
        self.current_frame = 0
        self.total_frames = 0
        self.stop_playback = False
        self.frame_buffer = None
        self.last_frame_time = 0
        self.is_muted = False
        self.volume = 100
        self.seeking = False
        
    def load_video(self, video_path):
        if self.cap is not None:
            self.stop_playback = True
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame = 0
        self.stop_playback = False
        self.is_playing = True
        self.play_pause_btn.configure(text="⏸")
        
        # Start playback thread
        threading.Thread(target=self.play_video, daemon=True).start()
        
    def play_video(self):
        try:
            while self.cap and not self.stop_playback:
                if not self.is_playing or self.seeking:
                    time.sleep(0.01)  # Reduced sleep time for better responsiveness
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    continue
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Update timeline without seeking
                if not self.seeking:
                    self.timeline.set(self.current_frame / self.total_frames * 100)
                
                # Update time display
                current_time = self.current_frame / self.fps
                total_time = self.total_frames / self.fps
                self.update_time_display(current_time, total_time)
                
                # Calculate frame delay for smooth playback
                frame_delay = 1 / (self.fps * self.playback_speed)
                
                # Display frame and wait for next frame
                self.display_frame(frame)
                time.sleep(max(0, frame_delay))
                
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            if self.cap:
                self.cap.release()
                
    def display_frame(self, frame):
        if frame is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate scaling
        scale = min(
            canvas_width / self.frame_width,
            canvas_height / self.frame_height
        )
        new_width = int(self.frame_width * scale)
        new_height = int(self.frame_height * scale)
        
        # Process frame
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(frame)
            
            # Center the frame
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference to prevent garbage collection
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
    def update_time_display(self, current_time, total_time):
        current_str = time.strftime('%M:%S', time.gmtime(current_time))
        total_str = time.strftime('%M:%S', time.gmtime(total_time))
        self.time_label.configure(text=f"{current_str} / {total_str}")
        
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.configure(text="⏸" if self.is_playing else "▶")
        
    def seek_video(self, value):
        if self.cap:
            self.seeking = True
            frame_num = int((value / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num
            self.seeking = False
            
    def restart_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.is_playing = True
            self.play_pause_btn.configure(text="⏸")
            
    def change_speed(self, speed):
        self.playback_speed = float(speed.replace("x", ""))
        
    def change_volume(self, value):
        self.volume = value
        if value == 0:
            self.volume_btn.configure(text="🔈")
        else:
            self.volume_btn.configure(text="🔊")
            
    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.volume_slider.set(0)
            self.volume_btn.configure(text="🔈")
        else:
            self.volume_slider.set(self.volume)
            self.volume_btn.configure(text="🔊")
            
    def cleanup(self):
        self.stop_playback = True
        if self.cap:
            self.cap.release()

class VideoRecognitionApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Smart Video Analyzer with Security Alerts")
        self.window.state('zoomed')
        ctk.set_appearance_mode("dark")
        
        # Define suspicious activities
        self.suspicious_activities = {
            'fighting': ['punch', 'fight', 'wrestle', 'kick', 'boxing', 'martial', 'combat', 'brawl'],
            'weapons': ['sword', 'knife', 'gun', 'weapon', 'axe', 'shooting'],
            'dangerous_behavior': ['jump', 'climb', 'parkour', 'dive', 'stunt', 'dangerous'],
            'substance_related': ['smoke', 'drink', 'alcohol', 'drug'],
            'property_damage': ['break', 'destroy', 'damage', 'vandal', 'smash']
        }
        
        # Add threshold for suspicious activity detection
        self.confidence_threshold = 0.15  # Lowered threshold to catch more potential threats
        
        # Add custom styling
        self.colors = {
            "primary": "#1f538d",
            "secondary": "#2d7dd2",
            "accent": "#45b7d1",
            "background": "#202020",
            "text": "#ffffff",
            "alert": "#ff4444"
        }
        
        self.setup_model()
        self.setup_ui()
        self.is_processing = False
        self.current_video = None
        self.alert_active = False

    def setup_model(self):
        self.device = "cpu"
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        
        # Load class names
        with open("kinetics_classnames.json", "r") as f:
            kinetics_classnames = json.load(f)
        
        self.kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            self.kinetics_id_to_classname[v] = str(k).replace('"', "")
        
        # Setup transform parameters
        self.num_frames = 32
        self.sampling_rate = 2
        self.frames_per_second = 30
        self.alpha = 4
        self.clip_duration = (self.num_frames * self.sampling_rate)/self.frames_per_second
        
        self.transform = self.get_transform()

    def setup_ui(self):
        # Create main container with grid layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_container.grid_columnconfigure(0, weight=3)  # Video section
        main_container.grid_columnconfigure(1, weight=2)  # Controls section
        
        # Left section - Video Display
        self.video_frame = ctk.CTkFrame(main_container)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Initialize EnhancedVideoPlayer
        self.video_player = EnhancedVideoPlayer(self.video_frame)
        
        # Right section - Controls and Results
        controls_frame = ctk.CTkFrame(main_container)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Title and Description
        ctk.CTkLabel(
            controls_frame,
            text="Smart Video Analyzer",
            font=("Helvetica", 24, "bold")
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            controls_frame,
            text="Upload a video to analyze actions and behaviors",
            font=("Helvetica", 14),
            text_color=self.colors["accent"]
        ).pack(pady=(0, 20))
        
        # Upload section
        upload_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        upload_frame.pack(fill="x", padx=20, pady=10)
        
        self.upload_button = ctk.CTkButton(
            upload_frame,
            text="Choose Video",
            command=self.upload_video,
            font=("Helvetica", 14, "bold"),
            height=40,
            fg_color=self.colors["primary"],
            hover_color=self.colors["secondary"]
        )
        self.upload_button.pack(side="left", padx=(0, 10))
        
        self.process_button = ctk.CTkButton(
            upload_frame,
            text="Start Analysis",
            command=self.start_processing,
            font=("Helvetica", 14, "bold"),
            height=40,
            fg_color=self.colors["secondary"],
            hover_color=self.colors["accent"],
            state="disabled"
        )
        self.process_button.pack(side="left")
        
        # File info section
        self.file_frame = ctk.CTkFrame(controls_frame)
        self.file_frame.pack(fill="x", padx=20, pady=10)
        
        self.path_label = ctk.CTkLabel(
            self.file_frame,
            text="No video selected",
            font=("Helvetica", 12)
        )
        self.path_label.pack(pady=10)
        
        # Progress section
        self.progress_frame = ctk.CTkFrame(controls_frame)
        self.progress_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready",
            font=("Helvetica", 12)
        )
        self.status_label.pack(pady=5)
        
        self.progress = ctk.CTkProgressBar(self.progress_frame)
        self.progress.pack(fill="x", pady=5)
        self.progress.set(0)
        
        # Results section
        self.results_frame = ctk.CTkFrame(controls_frame)
        self.results_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            self.results_frame,
            text="Analysis Results",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)
        
        self.result_label = ctk.CTkLabel(
            self.results_frame,
            text="No results yet",
            font=("Helvetica", 14)
        )
        self.result_label.pack(pady=5)
        
        # Confidence bars
        self.predictions_frame = ctk.CTkFrame(self.results_frame)
        self.predictions_frame.pack(fill="x", pady=10)
        
        self.confidence_bars = []
        self.confidence_labels = []
        for i in range(5):
            pred_frame = ctk.CTkFrame(self.predictions_frame, fg_color="transparent")
            pred_frame.pack(fill="x", pady=2)
            
            label = ctk.CTkLabel(
                pred_frame,
                text="",
                font=("Helvetica", 12)
            )
            label.pack(side="left", padx=5)
            self.confidence_labels.append(label)
            
            bar = ctk.CTkProgressBar(pred_frame)
            bar.pack(side="left", fill="x", expand=True, padx=5)
            bar.set(0)
            self.confidence_bars.append(bar)
        
        # Add Alert Section
        self.alert_frame = ctk.CTkFrame(self.results_frame)
        self.alert_frame.pack(fill="x", pady=10)
        
        self.alert_label = ctk.CTkLabel(
            self.alert_frame,
            text="Security Status: Normal",
            font=("Helvetica", 16, "bold"),
            text_color=self.colors["text"]
        )
        self.alert_label.pack(pady=5)
        
        # Add detailed alert text
        self.alert_details = ctk.CTkTextbox(
            self.alert_frame,
            height=100,
            width=300
        )
        self.alert_details.pack(pady=5)
        self.alert_details.insert("1.0", "No suspicious activities detected")
        self.alert_details.configure(state="disabled")

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.current_video = file_path
            self.video_player.load_video(file_path)
            self.path_label.configure(text=f"Selected: {Path(file_path).name}")
            self.process_button.configure(state="normal")
            self.status_label.configure(text="Ready to analyze")
            self.progress.set(0)

    def start_processing(self):
        if not self.is_processing and self.current_video:
            self.is_processing = True
            self.process_button.configure(state="disabled")
            self.upload_button.configure(state="disabled")
            self.status_label.configure(text="Processing video...")
            threading.Thread(target=self.process_video, args=(self.current_video,), daemon=True).start()

    def process_video(self, video_path):
        try:
            # Load video
            video = EncodedVideo.from_path(video_path)
            self.update_progress(0.2, "Loading video...")
            
            # Get clip
            video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)
            self.update_progress(0.4, "Extracting frames...")
            
            # Transform video
            video_data = self.transform(video_data)
            self.update_progress(0.6, "Analyzing frames...")
            
            # Get prediction
            inputs = video_data["video"]
            inputs = [i.to(self.device)[None, ...] for i in inputs]
            
            with torch.no_grad():
                preds = self.model(inputs)
            
            self.update_progress(0.8, "Generating results...")
            
            # Get top 5 predictions
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=5)
            
            # Update GUI with results
            self.window.after(0, self.update_results, pred_classes)
            
        except Exception as e:
            self.window.after(0, self.show_error, str(e))
        finally:
            self.is_processing = False
            self.window.after(0, self.reset_controls)

    def update_progress(self, value, status):
        self.window.after(0, lambda: self.progress.set(value))
        self.window.after(0, lambda: self.status_label.configure(text=status))

    def reset_controls(self):
        self.process_button.configure(state="normal")
        self.upload_button.configure(state="normal")
        self.status_label.configure(text="Analysis complete")

    def check_suspicious_activity(self, class_name, confidence):
        suspicious_found = []
        
        # Convert class name to lower case for comparison
        class_name_lower = class_name.lower()
        
        # Check each category of suspicious activities
        for category, activities in self.suspicious_activities.items():
            for activity in activities:
                # Use 'in' to check if activity keyword is part of the class name
                if activity.lower() in class_name_lower and confidence > self.confidence_threshold:
                    print(f"Found suspicious activity: {activity} in {class_name_lower} with confidence {confidence}")  # Debug print
                    suspicious_found.append({
                        'category': category,
                        'activity': class_name,
                        'confidence': confidence
                    })
                    break  # Break after finding first match in category
        
        return suspicious_found

    def update_results(self, predictions):
        indices = predictions.indices[0]
        values = predictions.values[0]
        
        # Reset alert state
        self.reset_alert()
        
        # Debug print predictions
        print("\nTop 5 Predictions:")
        
        # Check for suspicious activities
        all_suspicious_activities = []
        
        # Update confidence bars and check for suspicious activities
        for i in range(5):
            class_name = self.kinetics_id_to_classname[int(indices[i])]
            confidence = float(values[i])
            
            # Debug print each prediction
            print(f"Prediction {i+1}: {class_name} - Confidence: {confidence*100:.1f}%")
            
            # Check for suspicious activities
            suspicious = self.check_suspicious_activity(class_name, confidence)
            if suspicious:
                print(f"Suspicious activity found in prediction {i+1}: {suspicious}")
                all_suspicious_activities.extend(suspicious)
            
            # Update UI
            self.confidence_labels[i].configure(
                text=f"{class_name}: {confidence*100:.1f}%"
            )
            self.confidence_bars[i].set(confidence)
        
        # Update main prediction
        top_prediction = self.kinetics_id_to_classname[int(indices[0])]
        self.result_label.configure(
            text=f"Primary Action: {top_prediction}",
            font=("Helvetica", 16, "bold")
        )
        
        # Show alert if suspicious activities were found
        if all_suspicious_activities:
            print(f"\nTriggering alert for {len(all_suspicious_activities)} suspicious activities")
            self.show_alert(all_suspicious_activities)
        else:
            print("\nNo suspicious activities detected")
        
        self.progress.set(1.0)
        self.status_label.configure(text="Analysis complete!")

    def show_alert(self, suspicious_activities):
        if not suspicious_activities:  # Guard clause
            return
            
        self.alert_active = True
        self.alert_label.configure(
            text="⚠️ ALERT: Suspicious Activity Detected",
            text_color=self.colors["alert"]
        )
        
        # Prepare alert details with more formatting
        alert_text = "SUSPICIOUS ACTIVITIES DETECTED:\n" + "="*30 + "\n\n"
        
        for activity in suspicious_activities:
            alert_text += f"▶ Category: {activity['category'].replace('_', ' ').title()}\n"
            alert_text += f"▶ Activity: {activity['activity']}\n"
            alert_text += f"▶ Confidence: {activity['confidence']*100:.1f}%\n"
            alert_text += "-"*30 + "\n"
        
        # Update alert details in UI
        self.alert_details.configure(state="normal")
        self.alert_details.delete("1.0", "end")
        self.alert_details.insert("1.0", alert_text)
        self.alert_details.configure(state="disabled")
        
        # Show popup alert
        messagebox.showwarning(
            "⚠️ Security Alert",
            f"Suspicious activity detected in video!\n\n{alert_text}"
        )
        
        print(f"Alert shown with details:\n{alert_text}")  # Debug print

    def reset_alert(self):
        self.alert_active = False
        self.alert_label.configure(
            text="Security Status: Normal",
            text_color=self.colors["text"]
        )
        self.alert_details.configure(state="normal")
        self.alert_details.delete("1.0", "end")
        self.alert_details.insert("1.0", "No suspicious activities detected")
        self.alert_details.configure(state="disabled")

    def show_error(self, error_message):
        self.result_label.configure(
            text=f"Error: {error_message}",
            text_color="#ff4444"
        )
        self.status_label.configure(
            text="An error occurred",
            text_color="#ff4444"
        )
        self.progress.set(0)

    def get_transform(self):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256

        class PackPathway(torch.nn.Module):
            def __init__(self, alpha=4):
                super().__init__()
                self.alpha = alpha

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // self.alpha
                    ).long(),
                )
                return [slow_pathway, fast_pathway]

        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(self.num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway()
            ])
        )

    def on_closing(self):
        self.video_player.cleanup()
        self.window.destroy()

    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

if __name__ == "__main__":
    app = VideoRecognitionApp()
    app.run()