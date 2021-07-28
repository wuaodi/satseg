import json
import os, cv2
import numpy as np

train_json = 'E:\\study\\research\\1project_DA_classification\\satellite_seg\\dataset\\annotations\\instances_train2017.json'
train_path = 'E:\\study\\research\\1project_DA_classification\\satellite_seg\\dataset\\images\\train2017\\'
# 图片修改前保存路径，保存含有帆板的图片
save_path1 = 'E:\\study\\research\\1project_DA_classification\\satellite_seg\\datamake\\withcabin\\'
# 图片修改后保存路径，保存含有帆板，但是帆板被擦除后的图片
save_path2 = 'E:\\study\\research\\1project_DA_classification\\satellite_seg\\datamake\\withoutcabin\\'
# 图片修改后保存路径，保存只含有帆板区域的图片
save_path3 = 'E:\\study\\research\\1project_DA_classification\\satellite_seg\\datamake2\\onlysolar\\'

def visualization_bbox1(num_image, img_path, save_path1, save_path2, save_path3):# 需要画的第num副图片， 对应的json路径和图片路径

    image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
    id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

    image_path = os.path.join(img_path, str(image_name).zfill(5)) # 拼接图像路径
    save_path1 = os.path.join(save_path1, str(image_name).zfill(5)) # 拼接保存修改前图像路径
    save_path2 = os.path.join(save_path2, str(image_name).zfill(5))  # 拼接保存修改后图像路径
    save_path3 = os.path.join(save_path3, str(image_name).zfill(5))  # 拼接保存修改后图像路径
    print(save_path3)

    image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
    num_bbox = 0  # 统计一幅图片中bbox的数量

    shape = image.shape
    tempA = np.zeros(shape)


    for i in range(len(annotation_json['annotations'][::])):
        # 设置要读取的图片以及某一类 1帆板 2主体
        if  annotation_json['annotations'][i-1]['image_id'] == id \
                and annotation_json['annotations'][i-1]['category_id'] == 1:

            num_bbox = num_bbox + 1
            x, y, w, h = annotation_json['annotations'][i-1]['bbox']  # 读取边框


            ### 保存一：原图，保存含有某部件的图片
            # # 保存修改前的图片,把其他的保存代码部分注释掉，单独跑这个，能够得到包含某一类部件的照片原图
            # cv2.imwrite(save_path1, image)


            ### 保存二：修改，保存填充0或者首行像素的图片
            # 修改图片
            # # 方法一: 填充第一行像素
            # for i in range(int(h)):
            #     image[int(y)+i+1:int(y)+i+2, int(x):int(x)+int(w)] = \
            #         image[int(y)+i:int(y)+i+1, int(x):int(x)+int(w)]
            # cv2.imwrite(save_path2, image)
            # # 方法二: 填充0
            # image[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 0
            # cv2.imwrite(save_path2, image)
            # 显示标注框
            # image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)


            ### 保存三：修改，保存只保留bbox里面内容，其余部分全为0的图片
            tempA[int(y):int(y)+int(h), int(x):int(x)+int(w)] = \
                image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
            cv2.imwrite(save_path3, tempA)


    # print('The num_bbox of the display image is:', num_bbox)

    # 显示方式1：用plt.imshow()显示
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #绘制图像，将CV的BGR换成RGB
    # plt.show() #显示图像

    # 显示方式2：用cv2.imshow()显示
    # cv2.namedWindow(image_name, 0)  # 创建窗口
    # cv2.resizeWindow(image_name, 1000, 1000) # 创建500*500的窗口
    # cv2.imshow(image_name, image)
    # cv2.waitKey(0)

if __name__ == "__main__":
    with open(train_json) as annos:
        annotation_json = json.load(annos)
    print('the annotation_json num_key is:',len(annotation_json))  # 统计json文件的关键字长度
    print('the annotation_json key is:', annotation_json.keys()) # 读出json文件的关键字
    print('the annotation_json num_images is:', len(annotation_json['images'])) # json文件中包含的图片数量

    for i in range(len(annotation_json['images'])):
        # 调用定义的函数
        visualization_bbox1(i+1, train_path, save_path1, save_path2, save_path3)
        print(i+1)