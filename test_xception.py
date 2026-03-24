import cv2
from backend.frame_classifier import FrameClassifier

clf = FrameClassifier(model_type="xception_ffpp")

img = cv2.imread("real.jpg")

score = clf.classify(img)

print("Score:", score)
