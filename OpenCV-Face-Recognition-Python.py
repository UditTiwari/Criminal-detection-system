

import cv2

import os

import numpy as np




subjects = ["", "Ramiz Raja", "Elvis Presley", "Udit tiwari","criminals"]







def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);


    if (len(faces) == 0):
        return None, None


    (x, y, w, h) = faces[0]


    return gray[y:y + w, x:x + h], faces[0]





def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []

    labels = []


    for dir_name in dirs:


        if not dir_name.startswith("s"):
            continue;


        label = int(dir_name.replace("s", ""))


        subject_dir_path = data_folder_path + "/" + dir_name


        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;


            image_path = subject_dir_path + "/" + image_name


            image = cv2.imread(image_path)


            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)


            if face is not None:

                faces.append(face)

                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# I have defined a function that takes the path, where training subjects' folders are stored, as parameter. This function follows the same 4 prepare data substeps mentioned above. 
# 
# **(step-1)** On _line 8_ I am using `os.listdir` method to read names of all folders stored on path passed to function as parameter. On _line 10-13_ I am defining labels and faces vectors. 
# 
# **(step-2)** After that I traverse through all subjects' folder names and from each subject's folder name on _line 27_ I am extracting the label information. As folder names follow the `sLabel` naming convention so removing the  letter `s` from folder name will give us the label assigned to that subject. 
# 
# **(step-3)** On _line 34_, I read all the images names of of the current subject being traversed and on _line 39-66_ I traverse those images one by one. On _line 53-54_ I am using OpenCV's `imshow(window_title, image)` along with OpenCV's `waitKey(interval)` method to display the current image being traveresed. The `waitKey(interval)` method pauses the code flow for the given interval (milliseconds), I am using it with 100ms interval so that we can view the image window for 100ms. On _line 57_, I detect face from the current image being traversed. 
# 
# **(step-4)** On _line 62-66_, I add the detected face and label to their respective vectors.

# But a function can't do anything unless we call it on some data that it has to prepare, right? Don't worry, I have got data of two beautiful and famous celebrities. I am sure you will recognize them!
# 
# ![training-data](visualization/tom-shahrukh.png)
# 
# Let's call this function on images of these beautiful celebrities to prepare data for training of our Face Recognizer. Below is a simple code to do that.

# In[5]:

# let's first prepare our training data
# data will be in two lists of same size
# one list will contain all the faces
# and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# This was probably the boring part, right? Don't worry, the fun stuff is coming up next. It's time to train our own face recognizer so that once trained it can recognize new faces of the persons it was trained on. Read? Ok then let's train our face recognizer.

# ### Train Face Recognizer

# As we know, OpenCV comes equipped with three face recognizers.
# 
# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
# 
# I am going to use LBPH face recognizer but you can use any face recognizer of your choice. No matter which of the OpenCV's face recognizer you use the code will remain the same. You just have to change one line, the face recognizer initialization line given below. 

# In[6]:

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# or use EigenFaceRecognizer by replacing above line with
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# or use FisherFaceRecognizer by replacing above line with
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


# Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer. 

# In[7]:

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


# **Did you notice** that instead of passing `labels` vector directly to face recognizer I am first converting it to **numpy** array? This is because OpenCV expects labels vector to be a `numpy` array. 
# 
# Still not satisfied? Want to see some action? Next step is the real action, I promise! 

# ### Prediction

# Now comes my favorite part, the prediction part. This is where we actually get to see if our algorithm is actually recognizing our trained subjects's faces or not. We will take two test images of our celeberities, detect faces from each of them and then pass those faces to our trained face recognizer to see if it recognizes them. 
# 
# Below are some utility functions that we will use for drawing bounding box (rectangle) around face and putting celeberity name near the face bounding box. 

# In[8]:

# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# First function `draw_rectangle` draws a rectangle on image based on passed rectangle coordinates. It uses OpenCV's built in function `cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)` to draw rectangle. We will use it to draw a rectangle around the face detected in test image.
# 
# Second function `draw_text` uses OpenCV's built in function `cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)` to draw text on image. 
# 
# Now that we have the drawing functions, we just need to call the face recognizer's `predict(face)` method to test our face recognizer on test images. Following function does the prediction for us.

# In[9]:

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for.

# In[10]:

print("Predicting images...")

# load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

# perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
print("Prediction complete")

# display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
