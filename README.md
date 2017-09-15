# face_detect
Simple face detection implemented in Keras

First download FDDB face dataset and unzip, placing folders 2002, 2003 in this directory

To run:
# Build dataset of faces using jaccard scoring and ceil(score)
python distrib.py

# Train model (VGG, can be easily changed to Resnet-50)
python face_detect_keras.py

# Run on Camera image
python camera.py
