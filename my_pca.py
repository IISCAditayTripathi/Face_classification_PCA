import numpy
import pickle

class PrincleComponentAnalysis(object):
    """docstring for PrincleComponentAnalysis."""
    def __init__(self, data):
        super(PrincleComponentAnalysis, self).__init__()
        self.data = data

    def get_principle_components(self):
        sigma = 1/320*numpy.matmul(numpy.transpose(self.data), self.data)
        self.U, self.S, self.V = numpy.linalg.svd(sigma)

        return self.U, self.S, self.V

    def save_pc(self, name):
        if name == None:
            name = "princlple_components.pkl"
        data_dict = {'U':self.U, 'S': self.S, 'V': self.V}
        pickle.dump(data_dict, open(name, 'w'))

    def load_pc(self, name):
        if name == None:
            name = 'principle_components.pkl'
        data_dict = pickle.load(open(name, 'r'))
        return data_dict['U'], data_dict['S'], data_dict['V']
