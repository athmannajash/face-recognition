from tkinter import *
import sqlite3
import os
import cv2
import numpy as np
from PIL import Image

#Creating an instance of the class tkinter

root = Tk()
root.title('Machine')

detector=cv2.CascadeClassifier('classifier/haarface.xml')
cam=cv2.VideoCapture(0)

fname = StringVar()
lname = StringVar()
stdid= StringVar()
gender = StringVar()
def show():
    #print(fname.get())
    conn=sqlite3.connect("face.db")
    sqlid = "SELECT count(*) FROM user"
    cursor=conn.execute(sqlid)
    id=cursor.fetchone()[0]
    id = id + 1
    sql="SELECT count(*) FROM user WHERE STDID="+stdid.get()
    cursor=conn.execute(sql)
    data=cursor.fetchone()[0]
    if data ==0:
        params = (id, stdid.get(), fname.get(), lname.get(),gender.get())
        sql="INSERT INTO user values(?, ?, ?, ?,?);"
        conn.execute(sql, params)
        conn.commit()
    else:

        #-------------------------------------------------------------------
        #needs checking
        #====================================================
        sql="UPDATE user SET STDID = ? WHERE ID = ?;"
        conn.execute(sql,(stdid.get(), id))
        sql="UPDATE user SET FNAME = ? WHERE ID = ?;"
        conn.execute(sql,(fname.get(), id))
        sql="UPDATE user SET LNAME = ? WHERE ID = ?;"
        conn.execute(sql,(lname.get(), id))
        sql="UPDATE user SET GENDER = ? WHERE ID = ?;"
        conn.execute(sql,(gender.get(), id))
        #conn.execute(sql)
        conn.commit()
    conn.close()
    #directory where enrolled images will be stored
    folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset")
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
            cv2.rectangle(img,(x,y), (x+w,y+h),(255,255,255),2)
            cv2.waitKey(100)
        cv2.imshow("enrol face",img)
        cv2.waitKey(1)
        if (sampleNum>20):#20 snapshots of the new face are taken
            break
    cam.release()
    cv2.destroyAllWindows()
    os.system('python3 train.py')

#this creates entry boxes and buttons for enrolling
Label(root, width = 15, text="Reg No").grid(row=3, sticky=W)  #label
Entry(root,  textvariable = stdid).grid(row=3, column=2, sticky=E) #entry textbox
Label(root,width = 15, text="First Name").grid(row=5, sticky=W)  #label
Entry(root, textvariable = fname).grid(row=5, column=2, sticky=E) #entry textbox
Label(root, width = 15, text="Last Name").grid(row=7, sticky=W)  #label
Entry(root, textvariable = lname).grid(row=7, column=2, sticky=E) #entry textbox
Label(root, width = 15, text="Gender").grid(row=9, sticky=W)  #label
Entry(root, textvariable = gender).grid(row=9, column=2, sticky=E) #entry textbox
Button(root, text="Enroll", command=show).grid(row=11, column=0, sticky=W) #button


root.mainloop()
