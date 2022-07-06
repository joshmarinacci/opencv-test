
import cv2 as cv
import numpy as np

petsim = cv.CascadeClassifier('cascade/cascade.xml')
print("loaded classifier")


def findClickPositions(needle_img_path, haystack_img_path, threshold=0.5, debug_mode=None):
    print("loading",needle_img_path, haystack_img_path)
    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
#     needle_img = cv.imread(needle_img_path,   cv.IMREAD_UNCHANGED)
    print("detecting")
    rectangles = petsim.detectMultiScale(haystack_img)
    print("got rects",rectangles)

findClickPositions('pet-sim-needle.jpg','pet-sim-haystack.jpg', threshold=0.60, debug_mode='points')



# load the trained model
# loop_time = time()
# while(True):
    # get an updated image of the game
#     screenshot = wincap.get_screenshot()
    # do object detection
#     rectangles = cascade_limestone.detectMultiScale(screenshot)
    # draw the detection results onto the original image
#     detection_image = vision_limestone.draw_rectangles(screenshot, rectangles)
    # display the images
#     cv.imshow('Matches', detection_image)
