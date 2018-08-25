import numpy
import pickle
import numpy

class PrincleComponentAnalysis(object):
    """

    It finds the Princeiple components of a data array.
    Arguments:

    (numpy array)

    Available functions:

    get_principle_components(self): Performs the SVD on data array and returns U, S, V arrays.
    save_pc(string name): Save the U, S, V array as name.pkl
    load_pc(string name): Loads U, S, V array from name.pkl.
    reduce_image(array U, array image sample, int k): Performs the dimentionality reduction.
    reconstruct_image(array reduced_image, array reduced_U, mean): Reconstructs the oiginal images from low dim image vector.

    """

    def __init__(self, data):
        super(PrincleComponentAnalysis, self).__init__()
        self.data = data

    def get_principle_components(self):
        sigma = 1/320*numpy.matmul(numpy.transpose(self.data), self.data)
        self.U, self.S, self.V = numpy.linalg.svd(sigma)

        return self.U, self.S, self.V

    def save_pc(self, name='principle_components.pkl'):

        data_dict = {'U':self.U, 'S': self.S, 'V': self.V}
        pickle.dump(data_dict, open(name, 'wb'))

    def load_pc(self, name='principle_components.pkl'):

        data_dict = pickle.load(open(name, 'rb'))
        return numpy.asarray(data_dict['U']), numpy.asarray(data_dict['S']), numpy.asarray(data_dict['V'])

    def reduce_image(self, U, image_sample, k=40):
        reduced_U = U[:, 0:k]
        reduced_image = numpy.matmul(numpy.transpose(reduced_U), numpy.transpose(image_sample))

        return reduced_image, reduced_U

    def reconstruct_image(self, reduced_image, reduced_U, mean):
        reconstructed_image = numpy.matmul(reduced_U, reduced_image) + mean
        reshaped_image = reconstructed_image.reshape([112, 92])
        reshaped_image = numpy.asarray(reshaped_image, dtype=numpy.int16)

        return reshaped_image
