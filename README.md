# Training Mask RCNN model for weapon detection

## Running detection script:

For running script on image need to run `test_cv_image.py` file
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
- Remove code that checking GPU device:

    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    DEVICE = "/device:GPU:0"  # /CPU:0 or /device:GPU:0
    
    device_name = tf.test.gpu_device_name()
    if device_name != DEVICE:
        raise SystemError('GPU device not found')
    
    print('Found GPU at: {}'.format(device_name))
    ```
