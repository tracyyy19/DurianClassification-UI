from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import font

from PIL import Image, ImageOps, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize flag to track whether a new image is uploaded
new_image_uploaded = False

## function to select an image to process 
def upload_image():
    global my_image_re, new_image_uploaded, photo
    
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    
    if img_path:
        new_image_uploaded = True  # Set the flag to indicate a new image is uploaded
        
        img = Image.open(img_path)
        copy_img = img
        resized_image = copy_img.resize((250,250))
        photo = ImageTk.PhotoImage(resized_image)  # Keep a reference to the PhotoImage object
        selected_image_label.config(image=photo)
        
        my_image = ImageOps.fit(img, (128, 128))
        my_image_re = tf.keras.applications.vgg16.preprocess_input(np.array(my_image))

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img)
        axarr[0].set_title('Original Image')
        axarr[1].imshow(my_image)
        axarr[1].set_title('Resized Image')
        axarr[2].imshow(my_image_re)
        axarr[2].set_title('Preprocessed Image')

        plt.show()


from tkinter import messagebox

def show_probabilities(predictions):
    message = f"DURIAN SPECIES CALCULATED PROBABILITIES       \n\n"\
              f"Class 1: {predictions['class1']} - Probability: {predictions['prob1']:.2f}\n" \
              f"Class 2: {predictions['class2']} - Probability: {predictions['prob2']:.2f}\n" \
              f"Class 3: {predictions['class3']} - Probability: {predictions['prob3']:.2f}"
    messagebox.showinfo("Durian Species Probabilities", message)

def predictImage():
    global new_image_uploaded
    try:
        if new_image_uploaded:
            # Load the model and predict the image (change the path to the model accordingly)
            cnn_model = load_model('C:\\Users\\USER\\OneDrive\\Documents\\DurianClassificationGui\\durian_classification_trained_model.h5')
            cnn_model.run_eagerly = True
            probabilities = cnn_model.predict(np.array([my_image_re, ]), verbose=0)[0, :]
            number_to_class = ['D13 : Golden Pulp', 'D24: Sultan Durian', 'D197: Musang King']
            index = np.argsort(probabilities)
            predictions = {
                "class1": number_to_class[index[2]],
                "class2": number_to_class[index[1]],
                "class3": number_to_class[index[0]],
                "prob1": probabilities[index[2]],
                "prob2": probabilities[index[1]],
                "prob3": probabilities[index[0]],
            }

            # Find the class name with the highest probability
            max_probability_index = np.argmax(probabilities)
            class_name = number_to_class[max_probability_index]
            probability = probabilities[max_probability_index]

            # Display the predictions in the GUI
            prediction_text = f"Durian Class: {class_name}"
            img_label.config(text=prediction_text)
            show_probabilities(predictions)  # Display probabilities in a messagebox
            new_image_uploaded = False  # Reset the flag since the image has been classified
        else:
            img_label.config(text="Please upload a new image.")  # Inform user to upload a new image

    except NameError:
        img_label.config(text="Please select an image first.")
    except Exception:
        img_label.config(text="Error occurred. Please select a different image.")


# Create the main window
root = tk.Tk()
root.title("Durian Classification User Interface")
root.iconbitmap("durian-ui-logo.ico")

# Set fixed window size
window_width = 800
window_height = 500
root.minsize(window_width, window_height)

# Disable window resizing
root.resizable(False, False)

# Set window background color to white
root.configure(bg="white")

#display title 
image1 = Image.open("header-img.png")
photo1 = ImageTk.PhotoImage(image1)
label0 = Label(root, image=photo1, bg="white")
label0.pack()

#create a frame
frame1 = Frame(root, bg="white")
frame1.pack()

# Create a button for uploading an image
upload_button = Button(frame1, text="Upload an image", command=upload_image, width=20, bg="light yellow", fg="black", font=font.Font(size=14))
upload_button.grid(row=1, column=0, padx=20)

#display the durian catalog
image2 = Image.open("catalog-img.png")
photo2 = ImageTk.PhotoImage(image2)
label2 = Label(frame1, image=photo2, bg="white", foreground="black")
label2.grid(row=0, column=1, rowspan=4)

# Create a label for the selected image (initially empty)
selected_image_label = Label(frame1, bg="white")
selected_image_label.grid(row=0, column=0)

#create a prediction/classifier button 
# Create a button for uploading an image
classify_button = Button(frame1, text="Classify Durian",command=predictImage, width=20, bg="light yellow", fg="black", font=font.Font(size=14))
classify_button.grid(row=2, column=0, padx=20)

img_label = Label(frame1,bg="white",foreground="black", font=("Arial",12,"bold"))
img_label.grid(row=3, column=0)
 
# Run the Tkinter event loop
root.mainloop()
