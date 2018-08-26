import numpy
import sklearn
from skimage import io as IO
import random

class read_and_load(object):

    """Performs data read opeartion.
    Arguments:
    data_path(string array)

    Available methods:
    read_data(list datalist): reads images from the path list.
    make_train_test(): Make train-test splits and returns the train and test data arrays.
    sample_image(array image_array): Randomly samples an image from array of images.

    """
    def __init__(self, data_path):
        super(read_and_load, self).__init__()
        self.data_path = data_path

    def read_data(self, data_list):

        image_array_train = []
        image_array_valid = []
        image_class_train = []
        image_class_valid = []
        z = 0
        for data in data_list:
            z = z + 1
            for i in range(8):
                data_path = data[0:-1]+'/'+str(i+1)+'.pgm'
                im = IO.imread(data_path)
                im = [elements for rows in im for elements in rows ]
                image_array_train.append(im)
                image_class_train.append(z)
            for i in range(2):
                data_path = data[0:-1]+'/'+str(i+9)+'.pgm'
                im = IO.imread(data_path)
                im = [elements for rows in im for elements in rows ]
                image_array_valid.append(im)
                image_class_valid.append(z)
        image_array_train = numpy.asarray(image_array_train)
        mean_face_train = numpy.mean(image_array_train, axis=0)
        image_array_train = (image_array_train - mean_face_train)

        image_array_valid = numpy.asarray(image_array_valid)
        mean_face_valid = numpy.mean(image_array_valid, axis=0)
        image_array_valid = (image_array_train - mean_face_valid)

        return image_array_train, mean_face_train, image_class_train, image_array_valid, mean_face_valid, image_class_valid


    def make_train_test(self):

        data_list = open(self.data_path,'r')
        data_list = data_list.readlines()
        n_data = len(data_list)
        data_split = int(0.8 * n_data)
        train_data, train_mean, train_class, valid_data, valid_mean, valid_class = self.read_data(data_list)
        print("training set size:(%d,%d)"%(train_data.shape[0], train_data.shape[1]))
        # validation_data, valid_mean, valid_class = self.read_data(data_list[data_split:])
        print("validation set size:(%d,%d)"%(valid_data.shape[0], valid_data.shape[1]))
        return (train_data, train_mean, train_class, valid_data, valid_mean, valid_class)

    def sample_image(self, image_array):
        max = image_array.shape[0]
        sample = random.randint(1, max)
        return image_array[sample, :]
