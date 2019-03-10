#here we enroll users to the system
import cv2
import sqlite3
import os


detector=cv2.CascadeClassifier('classifier/haarface.xml')
cam=cv2.VideoCapture(0)

def insertOrUpdate(id,stdid,fname,lname,gender):
    conn=sqlite3.connect("face.db")
    sql="SELECT * FROM user WHERE ID="+str(id)
    cursor=conn.execute(sql)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
        if(isRecordExist==1):
            sql="UPDATE user SET STDID = ? WHERE ID = ?",(stdid, id)
            sql="UPDATE user SET FNAME = ? WHERE ID = ?",(fname, id)
            sql="UPDATE user SET LNAME = ? WHERE ID = ?",(lname, id)
            sql="UPDATE user SET GENDER = ? WHERE ID = ?",(gender, id)
            conn.execute(sql)
            conn.commit()

        else:
            params = (id, stdid, fname, lname, gender)
            conn.execute("INSERT INTO user(STDID, FNAME, LNAME, GENDER) VALUES(?, ?, ?, ?, ?)", params)
            #sql="INSERT INTO user(ID,STDID,FNAME,LNAME,GENDER) values("+str(id)+","+str(stdid)+","+str(fname)+","+str(lname)+","+str(gender)+")"
            #conn.execute(sql)
            conn.commit()
        conn.close()


id=input('enter user id')
stdid=input('enter user student id')
fname=input('enter your first name')
lname=input('enter your last name')
gender=input('enter your gender')

insertOrUpdate(id,stdid,fname,lname,gender)

folderName = fname                                                        # creating the person or user folder
folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/"+folderName)
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

sampleNum=0
while (True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite(folderPath + "/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow("enrol face",img)
    cv2.waitKey(1)
    if (sampleNum>20):#20 snapshots of the new face are taken
        break
cam.release()
cv2.destroyAllWindows()
