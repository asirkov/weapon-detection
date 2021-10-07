# Training Mask RCNN model for weapon detection

Data for training stored in Google Drive
https://drive.google.com/drive/folders/1-RxOIuePdoR9iBdt3hYbAtZACkFlKZZk

## Running detection script:

For running script on image need to run `test_cv_image_gpu.py` file
Parameters:
- `-w --weights`: required, relative path to weights file for MRCNN network
- `-i --in`: required, relative path to image file (files from URLs does not supports yet)
- `-o --out`: optional, the name of the source file, if it does not exist - will not be created. 
if the path contains some internal directories, they must exist, for example, for `/out/file1.jpg 'there must be a directory` / out'

Example:
```shell script
python -w logs/mask_rcnn_object_0010.h5 -i test/18e830f6c64b25a5090685a7b7ea3c041.jpg -o out/out6.jpg
```

## Running detection script on CPU:

For run script on CPU instead of GPU:

- Install `tensorflow` instead of `tensorflow-gpu` (also for suppress warnings change `tensorflow-gpu==1.15.2` to `tensorflow==1.15.2` in `requirements.txt`)
- Instead of `test_cv_image_gpu.py` run `test_cv_image_cpu.py`, it will not check the GPU device

Note: if you use `tensorflow-gpu` GPU will be used event if you run `test_cv_image_cpu.py` 
