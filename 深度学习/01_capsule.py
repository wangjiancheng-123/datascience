# 胶囊气泡、大小头等瑕疵检测
import cv2 as cv
import os
import numpy as np


def balance_check(img_path, img_file, im, im_gray):
    # 边沿提取
    # sobel = cv.Sobel(im_gray, cv.CV_64F,
    #                  1, 1, ksize=5)
    # cv.imshow("sobel", sobel)

    # lap = cv.Laplacian(im_gray, cv.CV_64F)
    # cv.imshow("lap", lap)

    # 模糊、膨胀去掉过细细节
    blurred = cv.GaussianBlur(im_gray, (5, 5), 0)
    kernel = np.ones((5,5), np.uint8)
    dilate = cv.dilate(blurred, kernel)
    cv.imshow("dilate", dilate)

    canny = cv.Canny(dilate, 60, 200)
    cv.imshow("canny", canny)

    # 在canny边沿检测图像上提取轮廓
    img, cnts, hie = cv.findContours(canny,
                                     cv.RETR_LIST,
                                     cv.CHAIN_APPROX_NONE)
    # 计算每个轮廓周长，并根据周长过滤
    new_cnts = [] # 存放经过筛选的轮廓列表
    if len(cnts) > 0:
        for c in cnts: # 遍历每个轮廓
            circle_len = cv.arcLength(c, True)
            #print("circle_len:", circle_len)

            if circle_len > 1000: # 过滤掉周长小于1000的轮廓
                new_cnts.append(c)
        # 对轮廓计算面积，并根据面积倒序排列
        new_cnts = sorted(new_cnts,
                          key=cv.contourArea,
                          reverse=True)
        new_cnts = new_cnts[1:2] # 切出面积第二大的轮廓
        # 绘制筛选后的轮廓
        im_cnt = cv.drawContours(im,
                                 new_cnts,
                                 -1,
                                 (0, 0, 255), 2)
        cv.imshow("im_cnt", im_cnt)

    # 求药丸轮廓中线位置
    max_x, max_y = new_cnts[0][0][0][0], new_cnts[0][0][0][1]
    min_x, min_y = max_x, max_y

    for cnt in new_cnts[0]:
        if cnt[0][0] >= max_x:
            max_x = cnt[0][0]
        if cnt[0][0] <= min_x:
            min_x = cnt[0][0]
        if cnt[0][1] >= max_y:
            max_y = cnt[0][1]
        if cnt[0][1] <= min_y:
            min_y = cnt[0][1]
    # 在原图上绘制直线
    #cv.line(im, (min_x, min_y), (max_x, min_y),
    #         (0, 0, 255), 2)
    # cv.line(im, (max_x, min_y), (max_x, max_y),
    #         (0, 0, 255), 2)
    # cv.line(im, (max_x, max_y), (min_x, max_y),
    #         (0, 0, 255), 2)
    # cv.line(im, (min_x, max_y), (min_x, min_y),
    #         (0, 0, 255), 2)

    # 计算药丸轮廓垂直方向中线
    mid_y = int((min_y + max_y) / 2)#中线y坐标
    # cv.line(im, (min_x, mid_y), (max_x, mid_y),
    #         (0, 0, 255), 2)

    mid_up = int((min_y + mid_y) / 2)
    mid_down = int((max_y + mid_y) / 2)

    cv.line(im, (min_x, mid_up), (max_x, mid_up),
            (0, 0, 255), 2)
    cv.line(im, (min_x, mid_down), (max_x, mid_down),
            (0, 0, 255), 2)

    #cv.imshow("im_line", im)

    # 求药丸轮廓和上中线、下中线的交点
    cross_point_up = set()
    cross_point_down = set()

    for cnt in new_cnts[0]: # 遍历药丸轮廓的每个点
        x, y = cnt[0][0], cnt[0][1]
        if y == mid_up:
            cross_point_up.add((x, y))
        if y == mid_down:
            cross_point_down.add((x, y))
    # 集合转列表
    cross_point_up = list(cross_point_up)
    cross_point_down = list(cross_point_down)

    for p in cross_point_up:
        cv.circle(im,
                  (p[0], p[1]), 8, #圆心、半径
                  (0,0,255), 2)
    for p in cross_point_down:
        cv.circle(im,
                  (p[0], p[1]), 8, #圆心、半径
                  (0,0,255), 2)
    cv.imshow("im_circle", im)

    # 求上中线、下中线长度
    len_up, len_down = 0, 0
    len_up = abs(cross_point_up[0][0] - cross_point_up[1][0])
    len_down = abs(cross_point_down[0][0] - cross_point_down[1][0])
    print("len_up:", len_up, " len_down:", len_down)

    if abs(len_up - len_down) > 8:
        print("大小头:", img_path)
    else:
        print("上下均衡:", img_path)

# 气泡检测函数
def bub_check(img_path, img_file, im, im_gray):
    # 二值化处理
    ret, im_bin = cv.threshold(im_gray,
                               170, 255,
                               cv.THRESH_BINARY)
    cv.imshow("im_bin", im_bin)

    # 腐蚀
    kernel = np.ones((1, 1), np.uint8)  # 计算核
    erosion = cv.erode(im_bin, kernel, iterations=3)
    cv.imshow("erosion", erosion)

    # 提取轮廓
    img, cnts, hie = cv.findContours(
        erosion,  # 图像
        cv.RETR_LIST,  # 检测所有轮廓
        cv.CHAIN_APPROX_NONE)  # 存储所有的轮廓坐标点

    new_cnts = [] # 记录筛选后的轮廓

    # 计算每个轮廓的面积，过滤掉过大、过小的轮廓
    if len(cnts) > 0 : # 检测到轮廓
        for cnt in cnts: # 遍历每个轮廓
            area = cv.contourArea(cnt) # 计算面积
            print("area:", area)
            if area < 10000 and area > 10:
                new_cnts.append(cnt)

    im_cnt = cv.drawContours(im,  # 在原图上绘制
                             new_cnts,  # 经过筛选轮廓数据
                             -1,  # 绘制所有轮廓
                             (0, 0, 255), 2)  # 轮廓颜色和粗细
    cv.imshow("im_cnt", im_cnt)

    if len(im_cnt) > 0:
        print("检测到气泡:", img_path)
        # 移动图像
        # ...

if __name__ == "__main__":
    # 先读取待检测图像
    img_dir = "test_img"  # 样本所在目录
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        # 拼接图像完整路径
        img_path = os.path.join(img_dir,
                                img_file)
        # 读取图像数据
        im = cv.imread(img_path, 1)
        im_gray = cv.cvtColor(im,
                              cv.COLOR_BGR2GRAY)
        cv.imshow("im", im)
        cv.imshow("im_gray", im_gray)

        # 调用函数气泡检测
        #bub_check(img_path, img_file, im, im_gray)

        # 调用函数做大小头检测
        balance_check(img_path, img_file, im, im_gray)

        cv.waitKey()
        cv.destroyAllWindows()
