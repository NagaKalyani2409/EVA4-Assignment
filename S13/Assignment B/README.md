YoloV3 Simplified for training on Colab with custom dataset.

A Collage of Training images

![github-small](https://github.com/NagaKalyani2409/EVA4-Assignment/blob/master/S13/Assignment%20B/train_batch0.png)
For custom dataset:

1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. Follow the installation steps as mentioned in the repo.
3. Download 500 images of your unique object.
4. Annotate the images using the Annotation tool.
5. We need to create custom.data file where classes = 1.
6. Similarly, the custom.txt file contains data in the following format : 
  ./data/customdata/images/img_001.jpg
  
7. We need to add custom.names file as you can see above. For our example, we downloaded images of Walle. Our custom.names file look        like this:
   Alex
   
5. Alex above will have a class index of 0.
6. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's    architecture.
7. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder.
8. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3
9. Search for 'classes=80' and change all three entries to 'classes=1

10. Since you are lazy (probably), you'll be working with very few samples. In such a case it is a good idea to change:
  burn_in to 100
  max_batches to 5000
  steps to 4000,4500

10. Run this command python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --         nosave
11. As you can see in the collage image above, a lot is going on, and if you are creating a set of say 500 images, you'd get a bonanza       of images via default augmentations being performed.
12. Click here to check out the video for where we played around annotating Alex in the clip
    https://www.youtube.com/watch?v=n9yn1vgXGlo

13. Results After training for 300 Epochs, results look awesome!

![github-small](https://github.com/NagaKalyani2409/EVA4-Assignment/blob/master/S13/Assignment%20B/test_img_001.jpg)
![github-small](https://github.com/NagaKalyani2409/EVA4-Assignment/blob/master/S13/Assignment%20B/test_img_004.jpg)
