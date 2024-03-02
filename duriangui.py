import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def selectImage():
    global my_image_re
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if img_path:
        img = Image.open(img_path)
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

def predictImage():
    try:
        # Load the model and predict the image (change the path to the model accordingly)
        cnn_model = load_model('C:\\Users\\USER\\OneDrive\\Documents\\DurianClassificationGui\\durian_classification_trained_model.h5')
        cnn_model.run_eagerly = True
        probabilities = cnn_model.predict(np.array([my_image_re, ]), verbose=0)[0, :]
        number_to_class = ['D13 (Red Prawn or Ang Hae)', 'D24 (Durian Sultan)', 'D197 (Musang King or Mao Shan Wang)']
        index = np.argsort(probabilities)
        predictions = {
            "class1": number_to_class[index[2]],
            "class2": number_to_class[index[1]],
            "class3": number_to_class[index[0]],
            "prob1": probabilities[index[2]],
            "prob2": probabilities[index[1]],
            "prob3": probabilities[index[0]],
        }
        print(predictions)

        # Display the predictions in the GUI
        prediction_text = f"Class 1: {predictions['class1']} ({predictions['prob1']:.2f})\n" \
                         f"Class 2: {predictions['class2']} ({predictions['prob2']:.2f})\n" \
                         f"Class 3: {predictions['class3']} ({predictions['prob3']:.2f})"
        img_label.config(text=prediction_text)
    except NameError:
        img_label.config(text="Please select an image first.")

    except Exception:
        img_label.config(text="Error occurred. Please select a different image.")

root = tk.Tk()
root.title("Image Processing")

select_button = tk.Button(root, text="Select Image", command=selectImage, padx=15, pady=10)
select_button.pack()

space_label = tk.Label(root, text="   ")
space_label.pack()

predict_button = tk.Button(root, text="Predict Image", command=predictImage, padx=15, pady=10)
predict_button.pack()

img_label = tk.Label(root)
img_label.pack()

root.mainloop()