from __future__ import division, print_function
import webbrowser
# coding=utf-8
import os
import numpy as np
from flask import request, render_template
from flask import Flask
from werkzeug.utils import secure_filename
import imagej
import os
ij = imagej.init()
#ij = imagej.init(['net.imagej:imagej:2.1.0', 'net.imagej:imagej-legacy'])
def calculate_intensity(file_path):
   # image_path='C:/Users/SANIA/Downloads/QA2.png'
    # Path to the image file
    image = ij.IJ.openImage(file_path)
    dataset = ij.py.to_dataset(image)

# Create a stats object for the dataset
    stats = ij.op().stats().mean(dataset)

# Retrieve the mean intensity value
    mean_intensity = stats.getRealDouble()
# Print the mean intensity value
    mean_intensity="Mean Intensity:", mean_intensity

# Close the image
    image.close()
    return mean_intensity
app = Flask(__name__)



@app.route("/")
def Predict():
    return render_template('predict.html')




@app.route('/', methods=['GET', 'POST'])
def display_intensity():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        intensity = calculate_intensity(file_path)
        
        intensity_str = str(intensity)
        
        # Return the intensity as a response
        return intensity_str
       

    return None


#app.run(debug=True)
if __name__ == '__main__':
    app.run(port=5001,debug=True)
