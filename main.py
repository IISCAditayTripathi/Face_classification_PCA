from read_data import read_and_load
import numpy
import pickle
from my_pca import PrincleComponentAnalysis as pca

file_path= '/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/face_image_paths.txt'
dataIO = read_and_load(file_path)

train_data, valid_data = dataIO.make_train_test()

princle_components = pca(train_data)

U, S, V = princle_components.get_principle_components()

# sigma = 1/320*numpy.matmul(numpy.transpose(train_data), train_data)
print(sigma.shape)

U, S, V = numpy.linalg.svd(sigma)
numpy.save('eigen_values.npy', S)
numpy.save('eigen_faces.npy', U)
# eigen_values = {'eigen':U}
# pickle.dump(eigen_values, open('eigen_values.pkl'))
