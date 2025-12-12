import cv2
import numpy as np

# تحميل نموذج HED من OpenCV (Pre-trained model)
proto_path = "deploy.prototxt"  # ملف البروتوتايب
model_path = "hed_pretrained_bsds.caffemodel"  # ملف النموذج

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # فلتر Canny
    edges_canny = cv2.Canny(gray, 100, 200)
    
    # فلتر HED
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(frame.shape[1], frame.shape[0]),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (frame.shape[1], frame.shape[0]))
    hed = (255 * hed).astype(np.uint8)
    
    # عرض الفيديوهات
    cv2.imshow('Original', frame)
    cv2.imshow('Canny', edges_canny)
    cv2.imshow('HED Model', hed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()