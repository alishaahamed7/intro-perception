"""Intentionally weak classical gate detector baseline.

Gate appearance (as seen facing the gate):
 
* left post:  black on top, red on the bottom
* right post: red on top, black on the bottom
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.types import GateEstimate

def blob_check(a, b):
    
    top_tol=0.3
    min_height_ratio = 0.5
    min_overlap=0.5

    # we're checking if 2 blobs look like the two posts of a gate
    ax = a[0]   # x; the left edge
    ay = a[1]   # y; the top edge
    aw = a[2]   # width
    ah = a[3]   # height

    bx = b[0]
    by = b[1]
    bw = b[2]
    bh = b[3]
    
    # this prevents a post and its reflection from being registered as a gate
    if min(ax + aw, bx + bw) > max(ax, bx):
        return False
    taller, shorter = max(ah, bh), min(ah, bh)
    
    # we want our posts to have similar height
    # sometimes the algo registers two "posts" but they appear in totally different locations in the picture
    # because they aren't actually a part of the gate
    if shorter / taller < min_height_ratio:
        return False
    
    # checking for the same issue here
    # sometimes 2 "posts" are found but they don't exist at the same height
    # therefore not a gate..
    if abs(ay - by) > top_tol * taller:
        return False

    # checking if the vertical ranges overlap
    overlap = min(ay + ah, by + bh) - max(ay, by)
    return overlap >= min_overlap * shorter


def best_pair(candidates, max_candidates=6):
    """Largest valid pair among the biggest candidates, ordered left to right."""
    top = sorted(candidates, key=lambda c: c[-1], reverse=True)[:max_candidates]
    best, best_area = [], 0.0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            area = top[i][-1] + top[j][-1]
            if blob_check(top[i], top[j]) and area > best_area:
                best, best_area = [top[i], top[j]], area
    return sorted(best, key=lambda c: c[0])

@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    # Find two vertical posts with fixed HSV thresholds.

    # ADDED: Default Slider Values
    def __init__(self):
        # so we're finding posts using color alone, but we're not taking lighting or brightness into consideration
        # Issues I am noticing:
        # - red looks different underwater, but the algo is only taking into consideration
        #    how red looks normally
        # - only checks brightness of black, doesn't take into consideration hues and saturation
        # - must be why the algo is detecting shadows, reflections, or other objects as gates
        super().__init__(
            red_h_min=((0, 180), 0),
            red_h_max=((0, 180), 10), # hue must be between 0 and 10
            red_s_min=((0, 255), 100), # saturation must be at least 100 out of 225
            red_v_min=((0, 255), 100), # value (how bright) must be at least 100 out of 225
            black_v_max=((0, 255), 110), # OG: black's brightness must be no more than 50 out of 225
                                            # I am changing it to 110 out of 225

        )

    
    # Understanding this:
    #    - frame is the photo, stored in an array as a grid of pixels
    #    - slider_vals - threshold settings, optional since the None
    
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):

        # if the picture doesn't exist, or it isn't loading properly, raises value error
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")


        # if no slider values were passed in, use 0
        if slider_vals is None:
            slider_vals = {}

        # Slider controls for HSV tuning
        r_h_min = slider_vals.get("red_h_min", 0)
        r_h_max = slider_vals.get("red_h_max", 10)
        r_s_min = slider_vals.get("red_s_min", 100)
        r_v_min = slider_vals.get("red_v_min", 100)
        
        b_v_max = slider_vals.get("black_v_max", 110) # initially 50, I am changing to 110
        
        # OpenCV --> BGR 
        # Now converting to HSV: hue (which color), saturation (how vivid), and value (how bright). 
        # HSV is easier for picking out “red” because the color is in one number.
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # if you're inside all 3 ranges you become white, outside you're black
        # so now we get a black and white picture called a mask
        red_mask = cv.inRange(
            hsv, 
            np.array((r_h_min, r_s_min, r_v_min)), 
            np.array((r_h_max, 255, 255))
        )
        
        # Broad black mask
        black_mask = cv.inRange(
            hsv, 
            np.array((0, 0, 0)), 
            np.array((180, 255, b_v_max))
        )

        combined_mask = cv.bitwise_or(red_mask, black_mask)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        contours = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        candidates = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            # Basic vertical aspect ratio check
            if area >= 30.0 and height >= max(10, int(width * 1.2)):
                candidates.append((x, y, width, height, area))
        
        # Sort by area and pick top 2
        candidates.sort(key=lambda c: c[-1], reverse=True)
        selected = best_pair(candidates)

        # default answer is no gate unless 2 posts were selected
        # IMPORTANT: if only 1 post is selected the algo still concludes no gate.
        estimate = GateEstimate.invisible()
        
        if len(selected) == 2:
            left_post, right_post = selected
            
            x1 = min(left_post[0], right_post[0])
            y1 = min(left_post[1], right_post[1])
            x2 = max(left_post[0] + left_post[2], right_post[0] + right_post[2])
            y2 = max(left_post[1] + left_post[3], right_post[1] + right_post[3])
 
            left_center = left_post[0] + left_post[2] / 2.0
            right_center = right_post[0] + right_post[2] / 2.0

            # This is the confidence score..
            separation = abs(right_center - left_center) / frame.shape[1]
            area_fraction = (left_post[-1] + right_post[-1]) / (frame.shape[0] * frame.shape[1])
            confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            
            estimate = GateEstimate(
                True,
                confidence,
                (left_center + right_center) / (2.0 * frame.shape[1]),
                (y1 + y2) / (2.0 * frame.shape[0]),
                x1 / frame.shape[1],
                y1 / frame.shape[0],
                (x2 - x1) / frame.shape[1],
                (y2 - y1) / frame.shape[0],
            ).normalized()
 
        if not debug:
            return estimate
            
        annotated = annotate(frame, estimate)
        for x, y, width, height, _ in selected:
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
        return estimate, [annotated, cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)]

def annotate(frame, estimate, label_prefix="classical"):
    canvas = frame.copy()
    if estimate.visible:
        height, width = frame.shape[:2]
        x1 = int(estimate.box_x * width)
        y1 = int(estimate.box_y * height)
        x2 = int((estimate.box_x + estimate.box_width) * width)
        y2 = int((estimate.box_y + estimate.box_height) * height)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{label_prefix}: gate {estimate.confidence:.2f}"
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2, cv.LINE_AA)
    else:
        cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return canvas
