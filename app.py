import sys
import os
from tkinter import *
import vehicle_detection_counting
import ambulance_detection

window=Tk()

def acc_detect():
    os.system('python accident.py --input videos/car_chase_01.mp4 --yolo yolo-coco')

window.title("Smart Traffic Management")

label1 = Label(window, text="Smart Traffic Management")
label1.grid(column=0, row=1)

label1 = Label(window, text="======================\n")
label1.grid(column=0, row=2)

label1 = Label(window, text="Ambulance Detection")
label1.grid(column=0, row=3)

btn = Button(window, text="Find Ambulance", bg="black", fg="white",command=ambulance_detection.amb_start)
btn.grid(column=0, row=4)

label1 = Label(window, text="Accident Detection")
label1.grid(column=0, row=5)

btn = Button(window, text="View Accident", bg="black", fg="white",command=acc_detect)
btn.grid(column=0, row=6)

label1 = Label(window, text="Dynamic Traffic Control")
label1.grid(column=0, row=7)

btn = Button(window, text="View Traffic", bg="black", fg="white",command=vehicle_detection_counting.vehicle_detect)
btn.grid(column=0, row=8)

window.mainloop()