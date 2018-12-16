#!/bin/bash
data=../cfg/rsna.data
test_cfg=../cfg/rsna_yolov3.cfg_test
weight=../backup/rsna_yolov3_18000.weights
if [ $1 = 1 ]
then
image_path=../images/401b69a0-3d07-47f8-bfea-4bc30aa51403.jpg
elif [ $1 = 2 ]
then
image_path=../images/f5926c42-623f-4a7d-9214-bf9f0964e9fd.jpg
elif [ $1 = 3 ]
then
image_path=../images/401b69a0-3d07-47f8-bfea-4bc30aa51403.jpg
elif [ $1 = 4 ]
then
image_path=../images/1794b7b4-de8c-479a-bfa7-2ab868c3061b.jpg
else
echo 'Your input is illegal. You should input "bash test.sh num". num is 1, 2, 3 or 4'
exit 0
fi
./darknet detector test $data $test_cfg $weight -thresh 0.05 $image_path
