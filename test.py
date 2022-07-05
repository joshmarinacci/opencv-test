import cv2 as cv
import numpy as np

# print(cv.getBuildInformation())

haystack_img = cv.imread('albion_farm.jpeg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpeg',   cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

threshold = 0.8

if max_val >= threshold:
    print('Found needle.')
else:
    print('needle not found.')

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]
top_left = max_loc
bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

print("needle w", needle_w)
cv.rectangle(haystack_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)

# print("got result",result)
# cv.imshow('Result',result)
# cv.waitKey()
# print("waited")

cv.imshow('Result',haystack_img)
cv.waitKey()
cv.imwrite('result.jpg',haystack_img)
