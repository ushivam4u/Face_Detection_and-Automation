## Defining Function for taking images for training model
import cv2,os
import numpy as np
import subprocess as sb
from os import listdir
from os.path import isfile, join

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
def Take_sample_img(Person_Name):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    try:
        # Collect 100 samples of your face from webcam input
        os.makedirs( "./faces/{}".format(Person_Name), 493 )
        while True:

            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save file in specified directory with unique name
        
                path = "./faces/{}".format(Person_Name)+"/"
                file_name_path = path + str(count) + '.jpg'.format(Person_Name)
                cv2.imwrite(file_name_path, face)

                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)

            else:
                #print("Face not found")
                pass

            if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()      
        print("Collecting Samples Complete")        
    except Exception as E :
        print ('Directory already  created. \nSample images already takens.{}'.format(E))
        
    
    
## Defining function for training the model
from os import listdir
from os.path import isfile, join
import joblib
def Train_model(Person_Name):
    data_path = './faces/{}'.format(Person_Name)+'/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    # model = cv2.face.createLBPHFaceRecognizer()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # pip install opencv-contrib-python
    # model = cv2.createLBPHFaceRecognizer()
    
    Trained_model  = cv2.face_LBPHFaceRecognizer.create()
    # Let's train our model 
    Trained_model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")

    #joblib.dump( Trained_model , '{}'.format(Person_Name)+'.pk1')
    Trained_model.save("{}".format(Person_Name)+".yml")
    
    
## Taking images for 1st Person
Take_sample_img("shivam")
Train_model("shivam")
    
    
## Taking images for 1st Person
Take_sample_img("Ronaldo")
Train_model("Ronaldo")


## Defining finction for predicting the face and giving required output
import cv2 
import numpy as np
def predict_face(person_one, person_two):
    rec1=cv2.face.LBPHFaceRecognizer_create()
    rec2=cv2.face.LBPHFaceRecognizer_create()
    rec1.read( "{}".format(person_one)+".yml")
    rec2.read( "{}".format(person_two)+".yml")
    

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def face_detector(img, size=0.5):

        # Convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return img, []


        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi


    # Open Webcam
    cap = cv2.VideoCapture(0)
    # vimal_model = keras.models.load_model('lbph_trained_data1.h5')
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Pass face to prediction model
            # "results" comprises of a tuple containing the label and the confidence value

            results1 = rec1.predict(face)
            
            results2 = rec2.predict(face)

            if results1[1] < 500 :
                confidence1 = int( 100 * (1 - (results1[1])/400) )
                confidence2 = int( 100 * (1 - (results2[1])/400) )
                display_string1 = str(confidence1) + '% Confident it is {}'.format(person_one)
                display_string2 = str(confidence2) + '% Confident it is {}'.format(person_two)
            #if results2[1] < 500:
                #confidence2 = int( 100 * (1 - (results2[1])/400) )
                #display_string2 = str(confidence) + '% Confident it is User2'

            

            if confidence1 > 80 :
                cv2.putText(image, display_string1, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                cv2.putText(image, "Hey {}".format(person_one)+", How r you!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                whatsapp()
                send_mail()
                break
            
                
        
            elif confidence2 > 80:
                
                cv2.putText(image, display_string2, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                cv2.putText(image, "Hey {}".format(person_two)+", How r you!", (300, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                aws_setup()
                break
                
            else:

                cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )

        except Exception as E:
            print(E)
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
  ## Asking model to analyze the face and do the required tasks
  predict_face("user3","user4")
  
  
