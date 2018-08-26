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

        image_array = []
        for data in data_list:
            for i in range(10):
                data_path = data[0:-1]+'/'+str(i+1)+'.pgm'
                im = IO.imread(data_path)
                im = [elements for rows in im for elements in rows ]
                image_array.append(im)
        image_array = numpy.asarray(image_array)
        mean_face = numpy.mean(image_array, axis=0)
        image_array = (image_array - mean_face)

        return image_array, mean_face


    def make_train_test(self):

        data_list = open(self.data_path,'r')
        data_list = data_list.readlines()
        n_data = len(data_list)
        data_split = int(0.8 * n_data)
        train_data, train_mean = self.read_data(data_list[0:data_split])
        print("training set size:(%d,%d)"%(train_data.shape[0], train_data.shape[1]))
        validation_data, valid_mean = self.read_data(data_list[data_split:])
        print("validation set size:(%d,%d)"%(validation_data.shape[0], validation_data.shape[1]))
        return (train_data, train_mean, validation_data, valid_mean)

    def sample_image(self, image_array):
        max = image_array.shape[0]
        sample = random.randint(1, max)
        return image_array[sample, :]
