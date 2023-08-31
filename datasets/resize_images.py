import cv2
import os

dir = "/storage/chendudai/data/hurba/dense/images"
out_dir = "/storage/hanibezalel/distilled-feature-fields/encoders/lseg_encoder/images/hurba"
os.chdir(dir)
for image_name in os.listdir(dir):
    image = cv2.imread(os.path.join(dir,image_name))
    resize_img = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
    cv2.imwrite(os.path.join(out_dir,image_name),resize_img)
    #pre, ext = os.path.splitext(image_name)
    #os.rename(image_name, pre + ".png")
    
    