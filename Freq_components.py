import cv2
import numpy as np
from scipy import fftpack

# Load the video path
cap = cv2.VideoCapture('/Users/terox/Downloads/test2.mp4')

# Define the lower and upper frequency limits
lower_freq = 0 #0hz
upper_freq = 1 #1hz

#Creating Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply 2D Fourier Transform to grayscaled frame 
    f_transform = fftpack.fft2(gray)
    f_shift = fftpack.fftshift(f_transform)

    # Generate frequency grid
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    f_shift[crow - lower_freq:crow + lower_freq, ccol - lower_freq:ccol + lower_freq] = 0
    f_shift[crow - upper_freq:crow + upper_freq, ccol - upper_freq:ccol + upper_freq] = 0
    f_ishift = fftpack.ifftshift(f_shift)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Display the output 
    cv2.imshow('Frequency Components (0-1 Hz)', img_back)
    cv2.imshow('original grayscaled video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()