import cv2
import numpy as np

image = cv2.imread("ornek_resim.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Hata: Resim bulunamadi!")
    exit()

h, w = image.shape

blurry_image = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)

kernel_dikey = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

kernel_yatay = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

opencv_dikey = cv2.filter2D(blurry_image, -1, kernel_dikey)
opencv_yatay = cv2.filter2D(blurry_image, -1, kernel_yatay)

manuel_dikey = np.zeros((h, w), dtype=np.uint8)
manuel_yatay = np.zeros((h, w), dtype=np.uint8)

blurry_float = blurry_image.astype(np.float32)

for y in range(1, h - 1):
    for x in range(1, w - 1):
        komsuluk = blurry_float[y - 1:y + 2, x - 1:x + 2]

        deger_dikey = np.sum(komsuluk * kernel_dikey)
        manuel_dikey[y, x] = int(np.clip(abs(deger_dikey), 0, 255))

        deger_yatay = np.sum(komsuluk * kernel_yatay)
        manuel_yatay[y, x] = int(np.clip(abs(deger_yatay), 0, 255))

cv2.imwrite("prewitt_dikey.jpg", manuel_dikey)
cv2.imwrite("prewitt_yatay.jpg", manuel_yatay)

cv2.imshow("1. Orijinal Resim", image)
cv2.imshow("2. Gauss Yumusatilmis", blurry_image)
cv2.imshow("3. Prewitt Dikey (Madde 1)", manuel_dikey)
cv2.imshow("4. Prewitt Yatay (Madde 2)", manuel_yatay)

cv2.waitKey(0)
cv2.destroyAllWindows()