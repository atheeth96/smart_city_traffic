# Smart City traffic management system
An python implementation of vehicle enumeration software.


# Project structure

 * [blob.py](blob.py) 
 * [Counter.py](Counter.py) 
 * [main.py](main.py)
 * [ObjectCounter.py](ObjectCounter.py)
 * [yolov3.cfg](yolov3.cfg)
 * [yolov3.txt](yolov3.txt)
 * [README.md](./README.md)

# Execution

 python main.py --video <input_video_path> --config yolov3.cfg --weights <yolo v3 weights> --classes yolov3.txt --output_video <output_video_path> <br />
 The yolo weights can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights)


# Results
 Sample Result

  ![Sample output](processed_video_2.1.gif)
  
# References

Nicholas Kajoh, ivy, (2019), GitHub repository, https://github.com/nicholaskajoh/ivy.git





 
