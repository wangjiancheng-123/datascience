# 01_fruits.py
# 利用CNN实现图像分类
# 数据集：爬虫从百度图片搜索结果爬取
# 内容：包含1036张水果图片
# 共5个类别（苹果288张、香蕉275张、葡萄216张、
#           橙子276张、梨251张）

################## 数据预处理 ##################
import os

name_dict = {"apple": 0, "banana": 1, "grape": 2,
             "orange": 3, "pear": 4}
data_root_path = "data/fruits/"  # 数据集所在目录
# 测试集、训练集文件路径
test_file_path = data_root_path + "test.txt"
train_file_path = data_root_path + "train.txt"
name_data_list = {}  # 记录每个类别有那些图片


def save_name_data_list(path,  # 图像路径
                        name):  # 类别名称
    if name not in name_data_list:  # 字典中没有该类别
        img_list = []  # 创建空列表
        img_list.append(path)  # 将图片存入列表
        name_data_list[name] = img_list  # 存入字典
    else:  # 字典中已经存在该类别
        name_data_list[name].append(path)


# 遍历数据集中的每个子目录，取出图像样本路径
# 并写入name_data_list字典
dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d  # 子目录完整路径
    # print(full_path)
    if os.path.isdir(full_path):  # 是一个目录
        imgs = os.listdir(full_path)  # 列出所有文件
        for img in imgs:
            img_full_path = full_path + "/" + img
            save_name_data_list(img_full_path,
                                d)  # 目录名称即类别名称
    else:  # 文件
        pass

# 遍历name_data_list字典，划分测试集、训练集
with open(test_file_path, "w") as f:
    pass

with open(train_file_path, "w") as f:
    pass

# 遍历字典
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 获取每个列别图片数量
    print("%s: %d张" % (name, num))

    for img in img_list:
        line = "%s\t%d\n" % (img, name_dict[name])
        if i % 10 == 0:  # 划分到测试集合
            with open(test_file_path, "a") as f:
                f.write(line)
        else:  # 划分到训练集
            with open(train_file_path, "a") as f:
                f.write(line)
        i += 1
print("数据预处理完成.")

############### 模型搭建/训练 ##################
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    """
    根据传入样本路径、类别，读取图像数据
    :param sample: 一行文本样本, 元组(文件路径,类别)
    :return: 返回图像数据、类别
    """
    img, label = sample  # img为路径, lable为类别
    if not os.path.exists(img):
        print(img, "文件不存在")

    # 读取文件内容
    img = paddle.dataset.image.load_image(img)
    # 将图像设置为固定大小
    img = paddle.dataset.image.simple_transform(
        im=img,  # 原始图像
        resize_size=100,  # 图像缩放大小
        crop_size=100,  # 裁剪图像大小
        is_color=True,  # 彩色图像
        is_train=True)  # 训练模型(做随机裁剪)
    # 归一化处理，将每个像素值转换为0~1之间
    img = img.astype("float32") / 255.0
    return img, label


# 从训练集中读取数据
def train_r(train_list, buffred_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                # 去除空格和换行符
                line = line.strip().replace("\n", "")
                img_path, lab = line.split("\t")

                yield img_path, int(lab)

    return paddle.reader.xmap_readers(
        train_mapper,  # 接收reader读取的数据二次处理
        reader,  # 原始读取器
        cpu_count(),  # 线程数量
        buffred_size)  # 缓冲区大小

# 定义reader
BATCH_SIZE = 32  # 批次大小

trainer_reader = train_r(train_list=train_file_path)
random_train_reader = paddle.reader.shuffle(
    reader=trainer_reader,
    buf_size=1300)  # 随机读取器
batch_train_reader = paddle.batch(
    random_train_reader,
    batch_size=BATCH_SIZE)

# 占位符
image = fluid.layers.data(name="image",
                          shape=[3, 100, 100],
                          dtype="float32")
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")

def create_CNN(image, type_size):
    """
    搭建卷积神经网络
    :param image: 图像数据(经过归一化处理)
    :param type_size:类别数量
    :return: 一组分类概率
    """
    # 第一组 conv/pool/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,  # 输入图像数据
        filter_size=3,  # 卷积核大小
        num_filters=32,  # 卷积核数量
        pool_size=2,  # 2*2区域做池化
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1,
                                dropout_prob=0.5)

    # 第二组 conv/pool/dropout
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 前一个dropout输出作为输入
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 2*2区域做池化
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2,
                                dropout_prob=0.5)

    # 第三组 conv/pool/dropout
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,  # 前一个dropout输出作为输入
        filter_size=3,  # 卷积核大小
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 2*2区域做池化
        pool_stride=2,  # 池化步长
        act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3,
                                dropout_prob=0.5)

    # fc
    fc = fluid.layers.fc(input=drop,
                         size=512,  # 神经元数量
                         act="relu")
    # dropout
    drop = fluid.layers.dropout(x=fc,
                                dropout_prob=0.5)
    # 输出层(使用softmax作为激活函数的fc)
    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict

# 创建VGG模型
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt, # 输入图像， 格式[N,C,H,W]
            pool_stride=2,#池化步长
            pool_size=2, #池化区域大小
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3, #卷积核大小
            conv_act="relu",#激活函数
            conv_with_batchnorm=True,#是否采用BN
            pool_type="max")#池化类型

    conv1 = conv_block(image, 64, 2, [0.0, 0.0])
    conv2 = conv_block(conv1, 128, 2, [0.0, 0.0])
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0.0])
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0.0])
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0.0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop,
                          size=512,
                          act=None)
    bn = fluid.layers.batch_norm(input=fc1,
                                 act="relu")#批量归一化
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)
    fc2 = fluid.layers.fc(input=drop2,
                          size=512,
                          act=None)
    predict = fluid.layers.fc(input=fc2,
                              size=type_size,
                              act="softmax")
    return predict


# 调用函数，创建模型
# predict = create_CNN(image=image, type_size=5)
predict = vgg_bn_drop(image=image, type_size=5)
# 损失函数
cost = fluid.layers.cross_entropy(
    input=predict,
    label=label)
avg_cost = fluid.layers.mean(cost)
# 准确率
accuracy = fluid.layers.accuracy(input=predict,
                                 label=label)
# 优化器
optimizer = fluid.optimizer.Adam(
    learning_rate=0.001)
optimizer.minimize(avg_cost)  # 优化目标函数

# 执行器
place = fluid.CUDAPlace(0)  # GPU训练
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder
feeder = fluid.DataFeeder(
    feed_list=[image, label],
    place=place)

costs = []  # 记录损失函数值
accs = []  # 记录准确度
times = 0
batchs = []  # 迭代次数

# 开始训练
for pass_id in range(5):
    train_cost = 0  # 临时变量，记录损失值
    train_acc = 0
    times += 1
    for batch_id, data in enumerate(batch_train_reader()):
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),  # 喂入参数
            fetch_list=[avg_cost, accuracy])
        # 打印损失值、准确率
        if batch_id % 20 == 0:
            print("pass_id:%d, batch_id:%d, cost:%f, acc:%f"
                  % (pass_id, batch_id,
                     train_cost[0], train_acc[0]))
            accs.append(train_acc[0])
            costs.append(train_cost[0])
            batchs.append(times)
# 保存模型
model_save_dir = "./model/fruits/"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(
    dirname=model_save_dir, #保存路径
    feeded_var_names=["image"],#预测时传入参数
    target_vars=[predict],#预测结果
    executor=exe)#执行器

print("模型保存成功:", model_save_dir)

# 训练过程可视化
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batchs, costs, color='red', label="Training Cost")
plt.plot(batchs, accs, color='green', label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()


#################### 预测 #####################
from PIL import Image

# 加载图像数据
def load_img(path): # path为图像路径
    img = paddle.dataset.image.load_and_transform(
        path, 100, 100, False).astype("float32")
    img = img / 255.0 # 归一化

    return img

# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place) #用于预测的执行器

infer_imgs = [] # 存放待预测的图像数据
test_img = "apple_1.png" # 待测试的图像
infer_imgs.append(load_img(test_img))#将图像数据存入待预测列表

infer_imgs = numpy.array(infer_imgs)#将列表转换为数组

# 加载模型
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir,
                                  infer_exe)
# 执行预测
results = infer_exe.run(infer_program,
                        feed={feed_target_names[0]:infer_imgs},
                        fetch_list=fetch_targets)
# print(results)

result = numpy.argmax(results[0][0])
for k, v in name_dict.items():
    if result == v:
        print("预测结果:", k)

# 显示待预测的图像
img = Image.open(test_img)
plt.imshow(img)
plt.show()










