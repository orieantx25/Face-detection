import cv2
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import os

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# Load the background image
background_image = Image.open('Resources/background.png')


# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(modePathList))
    

webcam_x, webcam_y = 50, 160 
webcam_w, webcam_h = 320, 240 

# Calculate the total window size based on the webcam and background dimensions
window_width = max(webcam_x + webcam_w, background_image.width)
window_height = max(webcam_y + webcam_h, background_image.height)

background_image[44:44 + 633, 808:808 + 414] = imgModeList[1]


# Initialize the Tkinter window
root = tk.Tk()
root.title("Webcam and Background")

# Create a Tkinter Canvas widget to display the video feed and background
canvas = Canvas(root, width=window_width, height=window_height)
canvas.pack()

def update_frame():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Convert the frame to a format that Tkinter can display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=img)

        # Update the Canvas with the webcam feed
        canvas.create_image(webcam_x, webcam_y, anchor=tk.NW, image=img)
        canvas.img = img

    # Schedule the next frame update
    canvas.after(10, update_frame)

# Start updating the frame
update_frame()

# Display the background image on the Canvas
background_image_tk = ImageTk.PhotoImage(background_image)
canvas.create_image(0, 0, anchor=tk.NW, image=background_image_tk)


close_button = tk.Button(root, text="Close", command=root.destroy)
close_button.pack()

root.mainloop()

cap.release()
