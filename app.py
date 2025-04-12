import torch
import json
import cv2
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
import time
from datetime import datetime
import threading
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

# PackPathway class remains the same...
class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // 4).long(),
        )
        return [slow_pathway, fast_pathway]

class VideoClassifier:
    def __init__(self):
        # Previous initialization code remains the same...
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        with open("kinetics_classnames.json", "r") as f:
            self.kinetics_id_to_classname = {v: k for k, v in json.load(f).items()}

        self.transform = self._create_transform()
        self.frame_buffer = []
        self.num_frames = 32
        self.frames_to_sample = 32
        
        self.fps = 0
        self.processing_time = 0
        self.last_prediction_time = time.time()
        self.frame_times = []
        
        self.setup_gui()
        
        self.is_running = False
        self.current_prediction = "No prediction yet"
        self.confidence_threshold = 0.3

    def _create_transform(self):
        # Transform creation remains the same...
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(32),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                ShortSideScale(size=256),
                CenterCropVideo(256),
                PackPathway()
            ])
        )

    def setup_gui(self):
        # Modern GUI setup with enhanced styling
        self.root = tk.Tk()
        self.root.title("Smart Activity Recognition")
        self.root.configure(bg='#1a1a1a')
        
        # Configure modern styles
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure custom styles
        style.configure("MainFrame.TFrame", background='#1a1a1a')
        style.configure("Card.TFrame", background='#2d2d2d')
        style.configure("Header.TLabel",
                       font=("Helvetica", 28, "bold"),
                       background='#1a1a1a',
                       foreground='#ffffff')
        style.configure("Prediction.TLabel",
                       font=("Helvetica", 24, "bold"),
                       background='#2d2d2d',
                       foreground='#4CAF50')
        style.configure("Stats.TLabel",
                       font=("Helvetica", 12),
                       background='#2d2d2d',
                       foreground='#9e9e9e')
        
        # Custom button styles
        style.configure("Start.TButton",
                       padding=10,
                       font=("Helvetica", 12, "bold"))
        style.configure("Stop.TButton",
                       padding=10,
                       font=("Helvetica", 12, "bold"))
        
        # Main container
        main_container = ttk.Frame(self.root, style="MainFrame.TFrame", padding=20)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Header
        header_label = ttk.Label(main_container,
                               text="Real-time Activity Recognition",
                               style="Header.TLabel")
        header_label.grid(row=0, column=0, pady=(0, 20))
        
        # Video card
        video_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        video_card.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Video display
        self.video_label = ttk.Label(video_card)
        self.video_label.grid(row=0, column=0, padx=2, pady=2)
        
        # Prediction card
        prediction_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        prediction_card.grid(row=2, column=0, sticky="ew", padx=5, pady=10)
        
        self.prediction_label = ttk.Label(prediction_card,
                                        text="Waiting for activity...",
                                        style="Prediction.TLabel",
                                        wraplength=600)
        self.prediction_label.grid(row=0, column=0, pady=5)
        
        # Stats card
        stats_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        stats_card.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Stats with icons (using Unicode characters as icons)
        self.fps_label = ttk.Label(stats_card,
                                  text="🔄 FPS: 0",
                                  style="Stats.TLabel")
        self.fps_label.grid(row=0, column=0, padx=20)
        
        self.processing_label = ttk.Label(stats_card,
                                        text="⚡ Processing: 0ms",
                                        style="Stats.TLabel")
        self.processing_label.grid(row=0, column=1, padx=20)
        
        # Control card
        control_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        control_card.grid(row=4, column=0, sticky="ew", padx=5, pady=10)
        
        # Buttons with icons
        self.start_button = ttk.Button(control_card,
                                     text="▶ Start Recognition",
                                     command=self.start_processing,
                                     style="Start.TButton",
                                     width=20)
        self.start_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = ttk.Button(control_card,
                                    text="⬛ Stop Recognition",
                                    command=self.stop_processing,
                                    style="Stop.TButton",
                                    state=tk.DISABLED,
                                    width=20)
        self.stop_button.grid(row=0, column=1, padx=10)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Center all content
        for widget in main_container.winfo_children():
            widget.grid_configure(sticky="n")
            
        # Set minimum window size
        self.root.minsize(800, 900)

    # The rest of the methods remain the same...
    def update_performance_metrics(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        one_second_ago = current_time - 1
        self.frame_times = [t for t in self.frame_times if t > one_second_ago]
        self.fps = len(self.frame_times)
        
        self.fps_label.config(text=f"🔄 FPS: {self.fps}")
        self.processing_label.config(text=f"⚡ Processing: {self.processing_time:.0f}ms")

    def process_frame(self, frame):
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) >= self.frames_to_sample:
            start_time = time.time()
            
            try:
                frames = torch.tensor(self.frame_buffer, device=self.device)
                frames = frames.permute(3, 0, 1, 2)
                
                video_data = self.transform({"video": frames})
                inputs = [i.to(self.device)[None, ...] for i in video_data["video"]]
                
                with torch.no_grad():
                    preds = torch.nn.Softmax(dim=1)(self.model(inputs))
                
                top_idx = preds.topk(1).indices[0][0]
                top_prob = preds.topk(1).values[0][0]
                
                if top_prob > self.confidence_threshold:
                    activity = self.kinetics_id_to_classname[int(top_idx)]
                    self.current_prediction = f"Detected Activity:\n{activity}\nConfidence: {top_prob*100:.1f}%"
                    self.prediction_label.config(text=self.current_prediction)
                
                self.processing_time = (time.time() - start_time) * 1000
                
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            self.frame_buffer = self.frame_buffer[-4:]

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.process_frame(frame_rgb)
                
                img = PIL.Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = PIL.ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                self.update_performance_metrics()
            
            self.root.after(10, self.update_frame)

    def start_processing(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.frame_buffer = []
            self.frame_times = []
            self.update_frame()
        else:
            self.prediction_label.config(text="Error: Could not open camera")

    def stop_processing(self):
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.prediction_label.config(text="Waiting for activity...")
        self.frame_buffer = []
        self.frame_times = []

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    classifier = VideoClassifier()
    classifier.run()