import cv2
import matplotlib.pyplot as plt
img = cv2.imread('pics/Lena.png')
# img = cv2.resize(img,(500,500))

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # مشتق افقی
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # مشتق عمودی
laplacian = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 20, 20)

cv2.imshow('canny',img)
# cv2.imshow('sobel_x',sobel_x)
# cv2.imshow('sobel_x',sobel_x)
# cv2.imshow('laplacian',laplacian)
# cv2.imshow('canny',canny)
cv2.waitKey(0)