import cv2 as cv
import numpy as np


def findClickPositions(needle_img_path, haystack_img_path, threshold=0.5, debug_mode=None):
    print("loading",needle_img_path, haystack_img_path)
    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(needle_img_path,   cv.IMREAD_UNCHANGED)
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    method = cv.TM_CCOEFF_NORMED

    result = cv.matchTemplate(haystack_img, needle_img, method)
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    print("locations",len(locations))

    print('found needle')
    line_color = (0,255,0)
    line_type = cv.LINE_4
    rectangles = []
    threshold = 0.17
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectangles.append(rect)
        rectangles.append(rect)
    print('grouping')
    rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
    print("grouped")
    points = []
    print("rect len",len(rectangles))
    if len(rectangles):
        for (x,y,w,h) in rectangles:
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                top_left = (x,y)
                bottom_right = (x+w,y+h)
                cv.rectangle(haystack_img, top_left, bottom_right,
                    color=line_color, lineType=line_type, thickness=2)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (center_x, center_y), color=(255,0,255), markerType=cv.MARKER_CROSS, markerSize=40, thickness=2)
        if debug_mode:
            cv.imshow('Matches', haystack_img)
            cv.waitKey()
    print("final points",points)
    return points


# findClickPositions('albion_cabbage.jpeg','albion_farm.jpeg', threshold=0.70, debug_mode='points')
findClickPositions('pet-sim-needle.jpg','pet-sim-haystack.jpg', threshold=0.60, debug_mode='points')
