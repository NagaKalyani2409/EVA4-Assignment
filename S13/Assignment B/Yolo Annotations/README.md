Structure: 

Images: folder where you have to throw in the pictures you want to annotate

Labels: automatically a respective yolo format self of the images' in the folder 
Images will be created in Labels with the same path. 

classes.txt: the different class of objects you want to detect using yolov3, 
write in the same syntax as in the one cloned directly from our repo.

main.py: GUI for annotations

process.py: creates txt files in yolov3 format

train.txt/test.txt: (or names similar to it, created from process.py) txt files in yolov3 format to be used during training too

In order to use the GUI and then create the files for the "orders" that receives the training for darknet: On terminal do the following:
    python main.py
    
