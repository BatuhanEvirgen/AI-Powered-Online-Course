import cv2
import imutils
import numpy as np
import threading
import sounddevice as sd
import wavio as wv
from flask import Flask, render_template, Response


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

  
def main():
    
    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    
    cam = cv2.VideoCapture('Demo video 3.mp4')

    while True:
        ret, frame = cam.read()
       
        frame = imutils.resize(frame, width=640, height=480)
        
        (H, W) = frame.shape[:2]
     
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        
        total_detections = detector.forward()

        for i in np.arange(0, total_detections.shape[2]):
            confidence = total_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(total_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = total_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                
                a = startX-80
                b = endX+80
                c = startY-40
                d = endY-40

                if d > 359 :
                    d = 360
                    c = 180
                if b > 639:
                    b  = 640
                    a = 387
                ##cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                pts1 = np.float32([[a,c],[b,c],[a,d],[b,d]])
                pts2 = np.float32([[0,0],[640,0],[0,360],[640,360]])

                M = cv2.getPerspectiveTransform(pts1,pts2)

                Nim = cv2.warpPerspective(frame,M,(640,360))

                cv2.imshow("Orjinal",frame)
                cv2.imshow("Islenmis",Nim) 
                
        
        frame = cv2.imencode('.jpg', Nim)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: Nim/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

main()

    
def seskayit():

    freq = 44100

    time = 10

    print("Kayit yapiliyor")
    recording = sd.rec(int(time * freq), 
                   samplerate=freq, channels=2)

    sd.wait()
    wv.write("seskayit.wav", recording, freq, sampwidth=2)
    print("kayit tamamlandi")

t1=threading.Thread(target=main,daemon=True)
t2=threading.Thread(target=seskayit)
t2.start()
t1.start()

@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
    