import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from USBCamera import USBCamera

class Dashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dashboard")
        self.root.attributes("-fullscreen", True)

        # Create a 3x3 grid
        self.frames = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            self.root.grid_rowconfigure(i, weight=1)
            self.root.grid_columnconfigure(i, weight=1)
            for j in range(3):
                frame = ttk.Frame(self.root, borderwidth=1, relief="solid")
                frame.grid(row=i, column=j, sticky="nsew")
                self.frames[i][j] = frame

        # Add USB camera feed to the top-left grid
        self.usb_cam_label = ttk.Label(self.frames[0][0])
        self.usb_cam_label.pack(expand=True, fill="both")

        self.usb_camera = USBCamera(camera_index=0, fps=30)
        self.update_usb_cam()

        # Add keybinding to exit fullscreen or close the app
        self.root.bind("<Escape>", self.exit_fullscreen)

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def update_usb_cam(self):
        try:
            frame = self.usb_camera.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.usb_cam_label.imgtk = imgtk
            self.usb_cam_label.configure(image=imgtk)
        except RuntimeError as e:
            print(f"Error capturing frame: {e}")
        self.root.after(10, self.update_usb_cam)

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.usb_camera.release()
        self.root.destroy()