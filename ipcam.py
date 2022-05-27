import cv2

# url = 'https://10.129.204.235:8080/video'
# cap = cv2.VideoCapture('https://10.129.204.235:8080/video')
cap = cv2.VideoCapture('rtsp://10.129.204.235:8080/h264_pcm.sdp')

while True:
    ret, image = cap.read()
    cv2.imshow('stream', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
