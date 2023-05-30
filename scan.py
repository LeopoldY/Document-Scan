# 导入必要的库
import cv2
import numpy as np
import pytesseract

# 导入PIL库
from PIL import Image

# 准备一张待处理图片文件
img = cv2.imread('receipt.jpg') # 读取一张购物小票的图片

# 自行设计算法实现文档边缘的检测
# 首先，将图像转换为灰度图，并进行高斯模糊
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为灰度图
blur = cv2.GaussianBlur(gray, (9, 9), 0) # 使用高斯模糊去除噪声

# 然后，使用Canny算法检测图像的边缘
edges = cv2.Canny(blur, 25, 50) # 使用Canny算法检测边缘，参数可以根据实际情况调整
cv2.imwrite('edges.jpg', edges)

# 接下来，使用findContours函数找到图像中的轮廓，并按面积降序排序
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 找到所有轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # 按面积降序排序，并只保留前5个轮廓

# 然后，遍历轮廓，寻找近似为四边形的轮廓，作为文档的边缘
for c in contours:
    peri = cv2.arcLength(c, True) # 计算轮廓的周长
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) # 使用多边形近似函数得到近似的多边形轮廓，参数可以根据实际情况调整
    if len(approx) == 4: # 如果近似的多边形轮廓有4个顶点，则认为是文档的边缘
        docCnt = approx # 保存文档的边缘
        break

# 最后，绘制文档的边缘，并显示结果
cv2.drawContours(img, [docCnt], -1, (0, 255, 0), 2) # 在原始图像上绘制文档的边缘，颜色为绿色，线宽为2像素
cv2.imwrite('detected.jpg', img)

# 将docCnt数组转换为numpy数组，并转换为浮点数类型
docCnt = np.array(docCnt, dtype="float32")

# 计算文档的宽度和高度
width = max(np.sqrt(((docCnt[0][0][0] - docCnt[1][0][0]) ** 2) + ((docCnt[0][0][1] - docCnt[1][0][1]) ** 2)),
            np.sqrt(((docCnt[2][0][0] - docCnt[3][0][0]) ** 2) + ((docCnt[2][0][1] - docCnt[3][0][1]) ** 2)))
height = max(np.sqrt(((docCnt[0][0][0] - docCnt[3][0][0]) ** 2) + ((docCnt[0][0][1] - docCnt[3][0][1]) ** 2)),
             np.sqrt(((docCnt[1][0][0] - docCnt[2][0][0]) ** 2) + ((docCnt[1][0][1] - docCnt[2][0][1]) ** 2)))

# 定义变换后的四个顶点坐标，按照左上，右上，右下，左下的顺序排列
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]], dtype="float32")

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(docCnt, dst)

# 对文档图像进行透视变换，得到校正后的图像
warped = cv2.warpPerspective(img, M, (int(width), int(height)))

# 图像顺时针旋转90度
warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

# 图像水平翻转
warped = cv2.flip(warped, 1)

cv2.imwrite('img-crop.jpg', warped)

# 读取文档图片
img = Image.open("img-crop.jpg")

# 使用pytesseract对图片进行OCR
text = pytesseract.image_to_string(img, lang="eng")

# 将输出结构保存到OCR-out.txt文件中
file = open("OCR-out.txt", 'w')
file.write(text)
file.close()