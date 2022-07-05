import cv2 as cv
import numpy as np
import pyautogui
from time import time, sleep

def findClickPositions(needle_img_path, haystack_img, threshold=0.5, debug_mode=None):
    print("loading",needle_img_path, haystack_img)
#     haystack_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)
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
    cv.imshow('Matches', haystack_img)
            #             cv.waitKey()
#     print("final points",points)
    return points




loop_time = time()
while(True):
    sleep(1)
    screenshot = pyautogui.screenshot(region=(0,0,800*2,650*2))
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
#     print("screenshot is",screenshot)
    points = findClickPositions('pet-sim-needle.jpg',screenshot, threshold=0.5, debug_mode='points')
#     print("poitns",points)
    if len(points) > 0:
        target = points[0]
        print("choosing point", target)
        pyautogui.moveTo(x=target[0]/2, y=target[1]/2)
        pyautogui.click()
        sleep(1)
#     cv.imshow('Computer Vision', screenshot)
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
