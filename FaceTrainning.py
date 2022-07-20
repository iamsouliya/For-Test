import os, cv2
import numpy as np
from PIL import Image

Recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "Image"


def getImagesID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert("L")
        faceNp = np.array(faceImg, "uint8")
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return IDs, faces


Ids, faces = getImagesID(path)
Recognizer.train(faces, np.array(Ids))
Recognizer.save("Data/trainingImage.yml")
cv2.destroyAllWindows()
