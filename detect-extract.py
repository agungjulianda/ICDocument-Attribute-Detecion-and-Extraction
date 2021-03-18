import csv
import cv2
import pytesseract


def pre_processing(image):
     
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return threshold_img 



def parse_text(threshold_img):

    
    tesseract_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT,config=tesseract_config)
    return details


def detection_attribute(image, threshold_img):
    faceCascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Face", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        crop_image = image[y:y + h, x:x + w]
        cv2.imwrite("FaceImageExtract.jpg", crop_image)
        
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 25)) 
    dilation = cv2.dilate(threshold_img, rect_kernel, iterations = 1) 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   
                                                     
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4500:
            x, y, w, h = cv2.boundingRect(cnt) 
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.putText(image, "Personal Data", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
   
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def format_text(details):
  
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word != '':
            word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)
            word_list = []

    return parse_text


def write_text(formatted_text):
    
    with open('extractionresult.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)

def print_data(filename):
	extraction_file = open (filename , "r")
	lines = extraction_file.readlines()
	for line in lines:
		print (line)
	extraction_file.close()

if __name__ == "__main__":
    
    image = cv2.imread('DocumentSample\\Doc2.jpg')
    resize_image = cv2.resize (image , (900,600), interpolation = cv2.INTER_AREA)
    
    thresholds_image = pre_processing(resize_image)
    
    parsed_data = parse_text(thresholds_image)
    
    accuracy_threshold = 5
    
    arranged_text = format_text(parsed_data)
    
    write_text(arranged_text)
    
    print_data("extractionresult.txt")

    detection_attribute(resize_image ,thresholds_image)
   
    