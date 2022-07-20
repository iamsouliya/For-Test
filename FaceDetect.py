from tkinter import *
import cv2
import pymysql


faceDetect = cv2.CascadeClassifier("Detect/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("Data/trainingImage.yml")
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (255, 0, 0)
db = "db_test"


def getProfile(Id):
    connection = pymysql.connect(
        host="localhost", user="root", password="", database=db
    )
    conn = connection.cursor()
    sql = "SELECT * FROM tb_face where f_Id='" + str(Id) + "';"
    conn.execute(sql)
    profile = None
    for row in conn:
        profile = row
    conn.close()
    return profile


while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        Id, conf = rec.predict(gray[y : y + h, x : x + w])
        if conf < 40:
            print(conf)
            global profile
            profile = getProfile(Id)
            if profile != None:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    str(profile[0]),
                    (x, y + h + 30),
                    fontface,
                    fontScale,
                    fontColor,
                )
        else:
            print("Unknown " + str(conf))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Unknown", (x, y + h + 30), fontface, fontScale, fontColor)
    cv2.imshow("Face", img)
    key = cv2.waitKey(1) & 0xFF == ord("q")
    if key:
        break
cam.release()
cv2.destroyAllWindows()
