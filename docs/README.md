# Building the Onnx version that we need

We need a specific version of onnx that is compatible with all of our network components (e.g. 2d average pooling). Build this onnx repo (branch resize):
https://github.com/fs-eire/onnxjs/tree/resize
Build Onnx and copy the complete onnxjs folder into the webmutex/mutexweb_demo directory.

# Running the example

To start the webserver locally you can use pythons build in http server. First `cd` into the webmutex/mutexweb_demo directory and run 
```python -m http.server 8000```

Then head to http://localhost:8000/ in your favorit browser. The page should show an example isbi image. Click on it to start the model inference + mws clustering. Note that this uses a bit of RAM and might freeze you machine for second.
