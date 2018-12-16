# Machine-Learning

1. Pneumonia detection is a [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) project.

2. Yolov3 model is forked from [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet).

3. Because the original image data is too large, I haven't added them here. You can download them in here. Input this in browser:  
`https://www.kaggle.com/c/10338/download-all` or use command:  
`kaggle competitions download -c rsna-pneumonia-detection-challenge`. 
I also deleted related images because they are too large. But you can still see the result. 

4. For processing the images please see the .ipynb there.

3. I train the model with initialized weight darknet53.conv.74 and get the best weight in 18000 iterations.
Download darknet weight by command:  
 `wget -q https://pjreddie.com/media/files/darknet53.conv.74`  
To download my weight in google drive, input `https://drive.google.com/open?id=1LMZKwe449B4x4sskQzQh2EgwOBVTYlTd` in browser and then download it,
and put it in backup directory.

4. To test, you can run 'bash test.sh num' in darknet directory, and num is from 1 to 4. 
1 and 2 are the right predictions in report. 3 and 4 are the wrong predictions.

Finally, please see the submission.ipynb in Yolo-test.