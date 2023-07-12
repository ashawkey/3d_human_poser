# A 3D human poser

It allows drag-editing of a simple 3D skeleton and rendering it into coco-style pose image.

https://user-images.githubusercontent.com/25863658/224222775-edb8c61f-b60c-4cf8-b59a-d6e8e3dbb49a.mp4

### Usge
```bash
pip install -r requirements.txt
# gui
python poser.py
```

* Left drag: rotate camera
* Middle click: pan camera
* Middle scroll: scale camera
* Right drag: select the nearest skeleton keypoint, and pan it.


```bash
# load saved pose
python poser.py --load poses/pose.json
# render all views
python poser.py --save rendered_poses
```
