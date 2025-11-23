### Aim

To implement Erosion and Dilation using Python and OpenCV.

### Software Required

1. Anaconda - Python 3.7
2. OpenCV

### Algorithm:

#### Step1:<br>
Import the necessary pacakages

#### Step2:<br>
Create the text using cv2.putText

#### Step3:<br>
Create the structuring element

#### Step4:<br>
Erode the image


#### Step5: <br>
Dilate the Image

 
### Program:
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_img():
    img =np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,text='HARI',org=(100,320), fontFace=font,fontScale= 4,color=(255,255,255),thickness=15,lineType=cv2.LINE_AA)
    return img
def display_img(img):
    plt.imshow(img,cmap='gray')
    plt.show()

img = load_img()
display_img(img)

#EROSION

kernel = np.ones((5, 5), dtype=np.uint8)
kernel
erosion1 = cv2.erode(img, kernel, iterations=1)
display_img(erosion1)

#DILATION

dilation=cv2.dilate(img,kernel,iterations=1)
display_img(dilation)
```
### OUTPUT

#### Input Image

![alt text](image-3.png)

#### Eroded Image

![alt text](image-4.png)

#### Dilated Image

![alt text](image-5.png)


## Result

Thus the generated text image is eroded and dilated using python and OpenCV.