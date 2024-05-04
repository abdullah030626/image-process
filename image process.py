import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import tkinter as tk
from tkinter import filedialog

def process_image():
    # Open an image
    image_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                            filetypes=(("Image files", ".jpg *.jpeg *.png *.bmp"), ("All files", ".*")))
    if not image_path:
        return
    
    image = cv.imread(image_path)

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Calculate histograms before adjustments
    r_hist_before = cv.calcHist([image_rgb], [0], None, [256], [0, 256])
    g_hist_before = cv.calcHist([image_rgb], [1], None, [256], [0, 256])
    b_hist_before = cv.calcHist([image_rgb], [2], None, [256], [0, 256])

    # Increase contrast
    enhancer_contrast = ImageEnhance.Contrast(Image.fromarray(image_rgb))
    image_contrast_increased = np.array(enhancer_contrast.enhance(1.5))

    r_hist_after = cv.calcHist([image_contrast_increased], [0], None, [256], [0, 256])
    g_hist_after = cv.calcHist([image_contrast_increased], [1], None, [256], [0, 256])
    b_hist_after = cv.calcHist([image_contrast_increased], [2], None, [256], [0, 256])

    # Reduce brightness
    enhancer_brightness = ImageEnhance.Brightness(Image.fromarray(image_contrast_increased))
    image_brightness_reduced = np.array(enhancer_brightness.enhance(0.5))

    r_hist_after2 = cv.calcHist([image_brightness_reduced], [0], None, [256], [0, 256])
    g_hist_after2 = cv.calcHist([image_brightness_reduced], [1], None, [256], [0, 256])
    b_hist_after2 = cv.calcHist([image_brightness_reduced], [2], None, [256], [0, 256])

    # Plot images and histograms
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 4, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')

    plt.subplot(3, 4, 2)
    plt.plot(r_hist_before, color='red')
    plt.title('Red Channel Histogram')

    plt.subplot(3, 4, 3)
    plt.plot(g_hist_before, color='green')
    plt.title('Green Channel Histogram')

    plt.subplot(3, 4, 4)
    plt.plot(b_hist_before, color='blue')
    plt.title('Blue Channel Histogram')

    plt.subplot(3, 4, 5)
    plt.imshow(image_contrast_increased)
    plt.title('Image After Contrast Increase')

    plt.subplot(3, 4, 6)
    plt.plot(r_hist_after, color='red')
    plt.title('Red Channel Histogram (After)')

    plt.subplot(3, 4, 7)
    plt.plot(g_hist_after, color='green')
    plt.title('Green Channel Histogram (After)')

    plt.subplot(3, 4, 8)
    plt.plot(b_hist_after, color='blue')
    plt.title('Blue Channel Histogram (After)')

    plt.subplot(3, 4, 9)
    plt.imshow(image_brightness_reduced)
    plt.title('Image After Brightness Reduction')

    plt.subplot(3, 4, 10)
    plt.plot(r_hist_after2, color='red')
    plt.title('Red Channel Histogram (After)')

    plt.subplot(3, 4, 11)
    plt.plot(g_hist_after2, color='green')
    plt.title('Green Channel Histogram (After)')

    plt.subplot(3, 4, 12)
    plt.plot(b_hist_after2, color='blue')
    plt.title('Blue Channel Histogram (After)')

    plt.tight_layout()
    plt.show()

# Create GUI
root = tk.Tk()
root.title("Image Processing")
root.geometry("200x50")

process_button = tk.Button(root, text="Process Image", command=process_image)
process_button.pack()

root.mainloop()