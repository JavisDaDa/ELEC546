from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
from linear_classifier import LinearSVM
import pandas as pd
from collections import defaultdict

# https://stackoverflow.com/questions/55535349/how-to-load-image-dataset-for-svm-image-classification-task
def load_image_files(container_path, dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if
    directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True,
            mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
             target=target,
             target_names=categories,
             images=images,
             DESCR=descr)


def get_train_test(train_path, test_path):
    image_dataset_train = load_image_files(train_path)

    image_dataset_test = load_image_files(test_path)
    X_train = image_dataset_train['data']
    y_train = image_dataset_train['target']
    X_test = image_dataset_test['data']
    y_test = image_dataset_test['target']
    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train, X_test, y_test):
    svm = LinearSVM()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-4, reg=5e4, num_iters=15000, verbose=True)
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss Value')
    plt.savefig('loss.png')
    plt.show()
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_test)
    train_acc = np.mean(y_train == y_train_pred)
    test_acc = np.mean(y_test == y_val_pred)
    print(f'training accuracy: {train_acc}')
    print(f'Test accuracy: {test_acc}')
    return y_val_pred, y_test


# def calculate_confusion(y_val_pred, y_test):
#     y_val_pred = list(y_val_pred)
#     y_test = list(y_test)
#     n = len(y_test)
#     matrix = [[0 for _ in range(3)] for _ in range(3)]
#     dic = defaultdict(int)
#     for i in range(n):
#         if y_test[i] == 0:
#             dic[0] += 1
#         elif y_test[i] == 1:
#             dic[1] += 1
#         else:
#             dic[2] += 1
#     for i in range(n):
#         if 0 <= i < dic[0] - 1:
#             pass
#         elif dic[0] <= i < dic[0] + dic[1] - 1:
#             pass
#         else:
#
#     pass

    # return df
def main():
    train_path = "D:\\Rice\\ELEC 546\\ELEC546\\Assignment7\\upload\Assignment07_data\\Assignment06_data\\Assignment06_data_reduced\\TrainingDataset"
    test_path = "D:\\Rice\\ELEC 546\\ELEC546\\Assignment7\\upload\Assignment07_data\\Assignment06_data\\Assignment06_data_reduced\\TestingDataset"
    X_train, y_train, X_test, y_test = get_train_test(train_path, test_path)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    y_val_pred, y_test = train_svm(X_train, y_train, X_test, y_test)



if __name__ == '__main__':
    main()
    confusion = [[0.3, 0.7, 0], [0.5, 0.4, 0.1], [0, 0, 1]]
    print(confusion)
