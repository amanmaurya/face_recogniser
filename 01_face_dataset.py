''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       
'''

import cv2
import os
import json


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
DATASET_DIR='dataset'
execution_path = os.getcwd()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
face_name = input('\n enter user Name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
f_path=os.path.join(execution_path,DATASET_DIR+'/'+str(face_id))

print(f_path)

count = 0

file_json_path='model.json'
a = []

entry = {'id': face_id,'name': face_name}

if not os.path.isfile(file_json_path):
    a.append(entry)
    with open(file_json_path, mode='w') as f:
        f.write(json.dumps(a, indent=2))
else:
    key='y'
    with open(file_json_path) as feedsjson:
        feeds = json.load(feedsjson)
    alredy=0    
    for x in feeds:
            if(x['id']==face_id):
                alredy=1
                break 
    if(alredy==1):
        key=input('\n ID Already Exists,for Retrain Press (Y), for abort press any key <return> ==>  ')
    else:
        feeds.append(entry)    

    if key.upper() != "Y":
        print('\n Aborting..........')
        exit(0)
   
    
    with open(file_json_path, mode='w') as f:
        f.write(json.dumps(feeds, indent=2))

if(os.path.exists(f_path) == False):
    os.mkdir(f_path)


while(True):

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1



        # Save the captured image into the datasets folder
        print(f_path,"/User.",str(count),".jpg")
        cv2.imwrite(f_path+"/User."+str(count)+".jpg", img)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


