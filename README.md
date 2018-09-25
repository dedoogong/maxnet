# maxnet
light version of 

make -j16

./maxnet detector test cfg/coco.data  cfg/yolov2.cfg yolov2.weights data/dog.jpg -i 1 cpu

./maxnet detector test cfg/coco.data  cfg/yolov2.cfg yolov2.weights data/dog.jpg -i 1 gpu

./maxnet detector test cfg/coco.data  cfg/yolov2.cfg yolov2.weights data/dog.jpg -i 1 cudnn
