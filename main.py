from read_data import read_and_load
import numpy
import pickle
from my_pca import PrincleComponentAnalysis as pca
from skimage import io as IO

file_path= '/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/face_image_paths.txt'
dataIO = read_and_load(file_path)

train_data, train_mean, valid_data, valid_mean = dataIO.make_train_test()


principle_components = pca(train_data)

# U, S, V = principle_components.get_principle_components()
#
# principle_components.save_pc()

U, S, V = principle_components.load_pc()


k = 40 # Number of PCA dimensions


train_sample = dataIO.sample_image(train_data)

reduced_image, reduced_U = principle_components.reduce_image(U, train_sample, k)

reconstruced_image = principle_components.reconstruct_image(reduced_image, reduced_U, train_mean)

original_image = train_sample + train_mean
original_reshaped = original_image.reshape([112, 92])
original_reshaped = numpy.asarray(original_reshaped, dtype=numpy.int16)
IO.imsave('original_reshaped_1.jpg', original_reshaped)
IO.imsave('reconstruced_image_1_k_40.jpg', reconstruced_image)
