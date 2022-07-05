import cv2 as cv
import numpy as np

haystack_img = cv.imread('albion_farm.jpeg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpeg',   cv.IMREAD_UNCHANGED)
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_SQDIFF_NORMED)
#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
# print('Best match top left position: %s' % str(max_loc))
# print('Best match confidence: %s' % max_val)

threshold = 0.17

locations = np.where(result <= threshold)
# print(locations)
locations = list(zip(*locations[::-1]))
print(locations)

if locations:
    print('found needle')
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    line_color = (0,255,0)
    line_type = cv.LINE_4

    for loc in locations:
#         print("location",loc)
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
        print("drawing at", top_left, bottom_right)
        cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)
    cv.imshow('Matches', haystack_img)
    cv.waitKey()
else:
    print("needle not found")

