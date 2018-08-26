# Dimensionality Reduction and face classification using PCA
This is a numpy based implementation of PCA for dimensionality reduction and image classification using K-Nearest-Neighbors.
- To run the program in default mode run:
```
python3 main.py
```
- To run in PCA mode(default mode):
```
python3 main.py --mode=pca
```
- To run in image reconstruction mode:
```
python3 main.py --mode=image_recons --nPC=30(your choice)
```
It will also save few reconstructed image samples.
- To run in image classification mode:
```
python3 main.py --mode=face_classification --nPC=30(your choice)
```

Please check other **flags** for more information.
