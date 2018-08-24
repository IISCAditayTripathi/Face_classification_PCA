import numpy
import sklearn
from skimage import io as IO

class read_and_load(object):
    """docstring for read_and_load."""
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
        return image_array


    def make_train_test(self):
        data_list = open(self.data_path,'r')
        data_list = data_list.readlines()
        n_data = len(data_list)
        data_split = int(0.8 * n_data)
        train_data = self.read_data(data_list[0:data_split])
        print("training set size:(%d,%d)"%(train_data.shape[0], train_data.shape[1]))
        validation_data = self.read_data(data_list[data_split:])
        print("validation set size:(%d,%d)"%(validation_data.shape[0], validation_data.shape[1]))
        return (train_data, validation_data)
