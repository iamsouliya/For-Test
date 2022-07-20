import cv2
import pymysql

faceDetect = cv2.CascadeClassifier("Detect/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(1)

db = "db_test"


def getProfile():
    connection = pymysql.connect(
        host="localhost", user="root", password="", database=db
    )
    conn = connection.cursor()
    sql = "SELECT F_Id FROM tb_face order by F_Id desc limit 1;"
    conn.execute(sql)
    profile = None
    for row in conn:
        profile = row
    connection.close()
    c = int("".join(map(str, profile)))
    return c


def insert(Id, Name):
    connection = pymysql.connect(
        host="localhost", user="root", password="", database=db
    )
    conn = connection.cursor()
    sql = "Select * from tb_face;"
    conn.execute(sql)
    # isRecordExist=0

    sql = (
        "Insert into tb_face(F_Id, Name) values('" + str(Id) + "','" + str(Name) + "');"
    )
    conn.execute(sql)
    connection.commit()
    conn.close()


oid = getProfile()
Id = oid + 1
Name = input("Enter your name:")
# Surname = input("Enter your surname:")
insert(Id, Name)
SampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        SampleNum = SampleNum + 1
        cv2.imwrite(
            "ImageData/ " + Name + "." + str(Id) + "." + str(SampleNum) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if SampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
