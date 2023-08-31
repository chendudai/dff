from colmap_utils import *
import cv2
import os 

#cv2.imread("")
#camdata = read_cameras_binary('/storage/hanibezalel/lerf/milan/sparse/0/cameras.bin')
#imdata = read_images_binary('/storage/hanibezalel/lerf/milan/sparse/0/images.bin')
camdata = read_cameras_binary('/storage/hanibezalel/colamap_data/sparse/0/cameras.bin')
imdata = read_images_binary('/storage/hanibezalel/colamap_data/sparse/0/images.bin')


id1 = 3
id2 = 5
#img1 = cv2.imread(os.path.join("/storage/hanibezalel/data_colmap/milan_data/images",imdata[id1][4]))
#img2 = cv2.imread(os.path.join("/storage/hanibezalel/data_colmap/milan_data/images",imdata[id2][4]))
img1 = cv2.imread(os.path.join("/storage/hanibezalel/colamap_data/images",imdata[id1][4]))
img2 = cv2.imread(os.path.join("/storage/hanibezalel/colamap_data/images",imdata[id2][4]))
#img1 = cv2.imread(os.path.join("/storage/hanibezalel/hurva_colmap/hurva/images",imdata[id1][4]))
#img2 = cv2.imread(os.path.join("/storage/hanibezalel/hurva_colmap/hurva/images",imdata[id2][4]))
kp1 = imdata[id1][5]
pointIds1= imdata[id1][6]
kp2 = imdata[id2][5]
pointIds2= imdata[id2][6]
matches,comm1,comm2 = np.intersect1d(imdata[id1][6],imdata[id2][6], return_indices=True)

kp1 = [(int(kp1[i][0]),int(kp1[i][1])) for i in comm1]
for point in kp1:
    cv2.circle(img1, point, 5,(0, 0,255),-1)
#pid1=np.argwhere(pointIds1==matches[1])[0][0]
#img1 = cv2.circle(img1,(int(kp1[pid1][0]),int(img1.shape[0]-1-kp1[pid1][1])),5,(0,0,255),-1)
    
kp2 = [(int(kp2[i][0]),int(kp2[i][1])) for i in comm2]
for point in kp2:
    cv2.circle(img2, point, 5,(0, 0,255),-1)

#pid2=np.argwhere(pointIds2==matches[1])[0][0]
#img2 = cv2.circle(img2,(int(kp2[pid2][0]),int(img1.shape[0]-1-kp2[pid2][1])),5,(0,0,255),-1)
    
cv2.imwrite("/storage/hanibezalel/Ha-NeRF/003_kp.jpg",img1)
cv2.imwrite("/storage/hanibezalel/Ha-NeRF/005_kp.jpg",img2)



print("hani")