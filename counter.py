import cv2

model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
window_name = "Head Counter"
font = cv2.FONT_HERSHEY_SIMPLEX
head_count = 0

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heads = model.detectMultiScale(gray, 1.1, 3)
    head_count += len(heads)
    cv2.putText(frame, "Head Count: {}".format(head_count), (10, 30), font, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()