This Document contains the updated files for tackling the Assignment named "Therapist and Child Detection and Tracking".
The file consists of :

1) Python Script for the code.
2) Sample output file in the form of .mp4 file.
3) Requirements.txt
4) Readme file
5) mmod_human_face_detector.dat file

To run the file, run the Requirements.txt file on the terminal using 'pip install -r requirements.txt' to check if the dependancies are satisfied in terms of the libraries. If not, install the necessary libraries.
Then, In the .py file or jupyter notebook, in the main section of the code, give the path of the input file and number of frame skips desired and run the file. Make sure that the dlib module is installed in your computer using the documentation:https://github.com/z-mahmud22/Dlib_Windows_Python3.x/tree/main.
An output file shall be seen in the same folder.

The approach uses YOLO for person detection,Haar face classifier for face detection and deepface for age estimation. The YOLO model is passed through the videos and the bounding boxes of the detected persons along with the id's are displayed on the output window.
The face coordinates are detected using Dlib's CNN Face Detector for face detections and deepface's age estimation is applied on this face to determine if the person is a therapist or a child and the corresponding bounding boxes and labels are printed.
The output is shown to the user and is saved as a .mp4 file.

Alternatively, the code is set up on Google Colab, the link: https://colab.research.google.com/drive/1_y1m7SXPiB1LDEEAcrAr6j9czqE4dc5k?usp=sharing

Note: This approach was seen to be very resource intensive and due to computation limitations, I wasn't able to test the code for all the video files but given the computation requiements are met, I am confident the code will run.
I hope you consider this solution.

Name: Gokul MVSS
mail: Mallemgokul@gmail.com

Applying for the role: Data Science (Computer Vision +Video LLMS) internship at CogniAble.
