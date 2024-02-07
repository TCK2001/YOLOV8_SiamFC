# YOLOV8_SiamFC
### First Step
+ Prepare the video
----
### Second Step
Run `demo.py` to generate the image (video -> image). It will take a long time, so be prepared. 
+ Uncomment below code (siamfc/ops.py) before running `demo.py`.
```python
roi = img[pt1[1]-30:pt2[1]+30, pt1[0]-30:pt2[0]+30]
    
# Before predicting, you have to change the data type to a NumPy array.
roi = np.array(roi)

# There are many hyperparameters in there. Please refer to the official website.
# https://docs.ultralytics.com/zh/usage/cfg/#predict
results = model.predict(roi, device='0', conf=0.25, max_det=1)
if results[0].masks is not None:
    clss = results[0].boxes.cls.cpu().tolist()
    masks = results[0].masks.xy

    annotator = Annotator(roi, line_width=2)

    for idx, (mask, cls) in enumerate(zip(masks, clss)):
        det_label = names[int(cls)]
        if det_label in objects_of_interest:
            annotator.seg_bbox(mask=mask,
                            det_label=det_label)

img[pt1[1]-30:pt2[1]+30, pt1[0]-30:pt2[0]+30] = roi
```
----
### Third Step
```shell
python demo.py --model pretrained --data video
```
----
# [After preparing all of the data]
----
### How to Run ?
```shell
python demo_image.py --model pretrained --data video\img
```
+ Checklist
  + You must check whether the pretrained `model` is in the `pretrained folder`, and also check the `evaluation images` in the folder.
----
### main part
```python
def track(self, img_files, box, visualize=False):
    frame_num = len(img_files)
    boxes = np.zeros((frame_num, 4))
    boxes[0] = box
    times = np.zeros(frame_num)

    for f, img_file in enumerate(img_files):
        img = ops.read_image(img_file)

        begin = time.time()
        if f == 0:
            self.init(img, box)
        else:
            boxes[f, :] = self.update(img)
        times[f] = time.time() - begin

        if visualize:
            ops.show_image(img, boxes[f, :])
            
    return boxes, times
```
+ Add this code to the top of the `ops.py` file.
```python
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
```
----
```python
# choose your yolov8 model
# v8n, v8s, v8m, v8l, v8x
model = YOLO("yolov8n-seg.pt")
model.to('cuda')
names = model.model.names

# select the object you want to segment/detect
objects_of_interest = ['bird']
```
`ops.show_image(img, boxes[f, :])` Track this code, then add the following code to line number 93 of `ops.py`.
```python
if visualize:
    # Select the roi size.
    roi = img[pt1[1]-30:pt2[1]+30, pt1[0]-30:pt2[0]+30]
    
    # Before predicting, you have to change the data type to a NumPy array.
    roi = np.array(roi)

    # There are many hyperparameters in there. Please refer to the official website.
    # https://docs.ultralytics.com/zh/usage/cfg/#predict
    results = model.predict(roi, device='0', conf=0.25, max_det=1)
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        annotator = Annotator(roi, line_width=2)

        for idx, (mask, cls) in enumerate(zip(masks, clss)):
            det_label = names[int(cls)]
            if det_label in objects_of_interest:
                annotator.seg_bbox(mask=mask,
                                det_label=det_label)

    img[pt1[1]-30:pt2[1]+30, pt1[0]-30:pt2[0]+30] = roi
    winname = 'window_{}'.format(fig_n)
    cv2.imshow(winname, img)
    cv2.waitKey(delay)
```
## Result
https://github.com/TCK2001/YOLOV8_SiamFC/assets/87925027/ad75ac9c-e598-497e-b0f3-2c3d49626a69
