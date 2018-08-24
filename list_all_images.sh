#!/usr/bin/env bash
root='/home/aditay/Matrix_Theory/Assignments/Assignment_2_PCA/assignment2_images'

for images in $(find $root -name *.pgm);do
  file_path=$(dirname $images)
  echo $file_path >> face_image_paths.txt
  # echo $images >> face_images.txt
done
