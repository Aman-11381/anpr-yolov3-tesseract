import cv2
import numpy as np
import re
import pytesseract

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

img = cv2.imread('car1.jpg')
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop = False)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y -h/2)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

########################## LABELLING THE DETECTED LICENSE PLATE USING RECTANGULAR BOX AND CONFIDENSE SCORE ############################

# for i in indexes.flatten():
#     x, y, w, h = boxes[i]
#     label = str(classes[class_ids[i]])
#     confidence = str(round(confidences[i],2))
#     color = colors[i]
#     cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#     cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

############################################# CROPPING THE REGION OF INTEREST #########################################################

crop_img = img[int(y)-5:int(y+h)+5, int(x)-5:int(x+w)+5]

################################################ RECOGNIZING THE PLATE ################################################

def extract_tesseract(img):
    resize_test_license_plate = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)
    gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)
    license_plate = pytesseract.image_to_string(gaussian_blur_license_plate, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')

    return license_plate

# extract_tesseract(crop_img)

def extract_tesseract2(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # Convert image to grayscale
    carplate_extract_img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply median blur
    carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray,3) # kernel size 3

    plate_num = pytesseract.image_to_string(carplate_extract_img_gray_blur, config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return plate_num

print(extract_tesseract2(crop_img, 150))

cv2.imshow('image', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()