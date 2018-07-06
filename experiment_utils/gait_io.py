from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# path: path to CASIA_B gei directory. This directory is result of copy_data script
# condition: one of five: 'bg1', 'bg2', 'nm', 'cl1', 'cl2'
def load_90_degree_gei_for_experiment1(src_dir, condition):
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []

    for i in range(1, 125):
        stri = ""
        if (i < 10):
            stri = "00" + str(i)
        elif (i < 100):
            stri = "0" + str(i)
        else:
            stri = str(i)
        dir = src_dir + stri + '/'
        # nm: first four of nm is used for train, last two is used for test
        if (condition == 'nm'):
            for j in range(1, 5):
                path = dir + stri + '-nm-' + '0' + str(j) + '-090.png'
                img = load_img(path, target_size=(140, 140))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:,:,0], axis=2)
                train_imgs.append(x)
                label = [0] * 124
                label[i - 1] = 1
                train_labels.append(label)
            for j in range(5, 7):
                path = dir + stri + '-nm-' + '0' + str(j) + '-090.png'
                img = load_img(path, target_size=(140, 140))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:, :, 0], axis=2)
                test_imgs.append(x)
                label = [0] * 124
                label[i - 1] = 1
                test_labels.append(label)
        # bg1: first image of bg is used for training, second is used for testing
        elif (condition == 'bg1'):
            train_path = dir + stri + '-bg-01-090.png'
            img = load_img(train_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            train_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            train_labels.append(label)

            test_path = dir + stri + '-bg-02-090.png'
            img = load_img(test_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            test_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            test_labels.append(label)
        # bg2: second image of bg is used for training, first is used for testing
        elif (condition == 'bg2'):
            train_path = dir + stri + '-bg-02-090.png'
            img = load_img(train_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            train_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            train_labels.append(label)

            test_path = dir + stri + '-bg-01-090.png'
            img = load_img(test_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            test_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            test_labels.append(label)
        # cl1: first image of cl is used for training, second is used for testing
        elif (condition == 'cl1'):
            train_path = dir + stri + '-cl-01-090.png'
            img = load_img(train_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            train_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            train_labels.append(label)

            test_path = dir + stri + '-cl-02-090.png'
            img = load_img(test_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            test_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            test_labels.append(label)
        # cl2: second image of cl is used for training, first is used for testing
        elif (condition == 'cl2'):
            train_path = dir + stri + '-cl-02-090.png'
            img = load_img(train_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            train_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            train_labels.append(label)

            test_path = dir + stri + '-cl-01-090.png'
            img = load_img(test_path, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            test_imgs.append(x)
            label = [0] * 124
            label[i - 1] = 1
            test_labels.append(label)

    return np.array(train_imgs), np.array(train_labels), np.array(test_imgs), np.array(test_labels)

if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_90_degree_gei_for_experiment1('Z:/DatasetB/GEI_CASIA_B/GEI_CASIA_B/gei90/', 'nm')
    print('done')