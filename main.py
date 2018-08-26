from read_data import read_and_load
import numpy
import pickle
from my_pca import PrincleComponentAnalysis as pca
from skimage import io as IO
from sklearn.neighbors import KNeighborsClassifier

file_path= '/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/face_image_paths.txt'
dataIO = read_and_load(file_path)

train_data, train_mean, train_class, valid_data, valid_mean, valid_class = dataIO.make_train_test()


bias = train_mean - valid_mean


principle_components = pca(train_data)

U, S, V = principle_components.get_principle_components()

principle_components.save_pc()

U, S, V = principle_components.load_pc()


k = 200 # Number of PCA dimensions

reduced_U = numpy.transpose(U[:, 0:k])
reduced_images_train = numpy.matmul(reduced_U, numpy.transpose(train_data))

reduced_images_valid = numpy.matmul(reduced_U, numpy.transpose(valid_data))

neigh = KNeighborsClassifier(n_neighbors=8)

neigh.fit(numpy.transpose(reduced_images_train), numpy.asarray(train_class))
predictions = neigh.predict(numpy.transpose(reduced_images_valid))
print(predictions)
'''
train_sample = dataIO.sample_image(train_data)
test_sample = valid_data[1,:]

reduced_image, reduced_U = principle_components.reduce_image(U, test_sample, k)

reconstruced_image = principle_components.reconstruct_image(reduced_image, reduced_U, valid_mean) + 90

original_image = test_sample + valid_mean - bias
test_mean_face = numpy.asarray(valid_mean.reshape([112, 92]), dtype=numpy.int16)

original_reshaped = original_image.reshape([112, 92])
original_reshaped = numpy.asarray(original_reshaped, dtype=numpy.int16)

IO.imsave('test_original_reshaped_1.jpg', original_reshaped)
IO.imsave('test_reconstruced_image_1_k_30.jpg', reconstruced_image)
IO.imsave('test_mean_face.jpg', test_mean_face)
'''
