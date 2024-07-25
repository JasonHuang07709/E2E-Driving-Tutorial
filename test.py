import pickle
import numpy as np
import cv2

def loadFromPickle():
    with open("features_40", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels

def reshape_and_display_image(features):
    # 原始图像形状
    frame_shape = (160, 320, 3)

    # 将 gray 调整回原始图像的大小
    resized_gray = cv2.resize(features[0], (frame_shape[1], frame_shape[0]))

    # 创建一个空的三通道图像
    reconstructed_frame = np.zeros(frame_shape, dtype=np.uint8)

    # 将 resized_gray 复制到 reconstructed_frame 的每个通道
    reconstructed_frame[:, :, 0] = resized_gray  # 这里将灰度图像放到第一个通道
    reconstructed_frame[:, :, 1] = resized_gray  # 如果需要，可以将其放到第二个或第三个通道
    reconstructed_frame[:, :, 2] = resized_gray

    # 显示结果（可选）
    cv2.imshow("Reconstructed Frame", reconstructed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    features, labels = loadFromPickle()

    # 打印加载的数据的形状，以验证数据是否正确加载
    print(f'Features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    # 重塑并显示图像
    reshape_and_display_image(features)


    # 实验证明，对于数据集的处理是如下：
    '''
    1. 原图是（160, 320, 3）通道的RGB图像
    2. 先把图像转到HSV图像并且只取[:,:,1]这个通道数值然后再resize到（40,40）
    3. 然后把所有的图像放到一个数组中保存为features
    '''
