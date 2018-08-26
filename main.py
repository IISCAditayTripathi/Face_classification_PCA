from read_data import read_and_load
import numpy
import pickle
from my_pca import PrincleComponentAnalysis as pca
from skimage import io as IO
from sklearn.neighbors import KNeighborsClassifier
import argparse

parser = argparse.ArgumentParser(description='Main script to perform PCA and face classification')

parser.add_argument('--data_path', type=str, default='/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/face_image_paths.txt',
                    help='path to the text file containing images paths')
parser.add_argument('--eigen_path', type=str, default='/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/principle_components.pkl',
                    help='path to the file containing eigen vectors')
parser.add_argument('--mode', type=str, default='pca',
                    help='Specify the mode: pca, image_recons, face_classification')
parser.add_argument('--nPC', type=int, default=40,
                    help='Number of principle components (default: 40)')

args = parser.parse_args()
file_path = args.data_path

# file_path= '/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/code/face_image_paths.txt'
dataIO = read_and_load(file_path)

train_data, train_mean, train_class, valid_data, valid_mean, valid_class = dataIO.make_train_test()
bias = train_mean - valid_mean

principle_components = pca(train_data)
k = args.nPC # Number of PCA dimensions
if args.mode == 'pca':
    U, S, V = principle_components.get_principle_components()
    principle_components.save_pc(args.eigen_path)

if args.mode == 'face_classification':
    U, S, V = principle_components.load_pc(args.eigen_path)

    reduced_U = numpy.transpose(U[:, 0:k])
    reduced_images_train = numpy.matmul(reduced_U, numpy.transpose(train_data))

    reduced_images_valid = numpy.matmul(reduced_U, numpy.transpose(valid_data))

    neigh = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

    neigh.fit(numpy.transpose(reduced_images_train), numpy.asarray(train_class))
    predictions = neigh.predict(numpy.transpose(reduced_images_valid))
    predictions = predictions.tolist()
    accurate = 0
    for i in range(len(predictions)):
        if predictions[i] == valid_class[i]:
            accurate += 1
    print("face classification accuracy for k=%d is: %d/%d = %f"%(k, accurate, len(predictions), float(accurate)/(len(predictions))))

if args.mode == 'image_recons':
    U, S, V = principle_components.load_pc(args.eigen_path)
    train_sample = dataIO.sample_image(train_data)
    test_sample = valid_data[1,:]
    reduced_image, reduced_U = principle_components.reduce_image(U, test_sample, k)

    reconstruced_image = principle_components.reconstruct_image(reduced_image, reduced_U, valid_mean)

    original_image = test_sample + valid_mean - bias
    test_mean_face = numpy.asarray(valid_mean.reshape([112, 92]), dtype=numpy.int16)

    original_reshaped = original_image.reshape([112, 92])
    original_reshaped = numpy.asarray(original_reshaped, dtype=numpy.int16)

    IO.imsave('test_original_reshaped.jpg', original_reshaped)
    IO.imsave('test_reconstruced_image_k_'+str(k)+'.jpg', reconstruced_image)
    IO.imsave('test_mean_face.jpg', test_mean_face)
