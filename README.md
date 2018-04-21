# TensorRT-SSD
Use TensorRT API to implement Caffe-SSD， SSD（channel pruning）， Mobilenet-SSD

============================================
I hope my code will help you learn and understand the TensorRT API better. It’s welcome to discuss the deep learning algorithm, model optimization， TensorRT API and so on， and learn from each other.
==============================================================================

#Introduction:

1. The original Caffe-SSD can run 3-5fps on my jetson tx2.
2. TensorRT-SSD can run 8-10fps on my jetson tx2.
3. TensorRT-SSD(channel pruning) can run 16-17fps on my jetson tx2.
4. TensorRT-Mobilenet-SSD can run 40-43fps on my jetson tx2（it‘s cool！）， and run 100+fps on gtx1060.

#Requirements:

1. TensorRT3.0
2. Cuda8.0 or Cuda9.0
3. OpenCV


The code will be published shortly...

==============================================

In the Other_layer_tensorRT folder, there are the implementation of some other layers with TensorRT api， including：

1. PReLU

Continuously updated...

1. 2018/02/06, update detection_out layer
2. 2018/03/07， add the common.cpp file
3. 2018/04/21， TensorFlow 1.7 wheel with JetPack 3.2.（enable TensorRT support)

  Python2.7：https://nvidia.app.box.com/v/TF170-py27-wTRT

  Python3.5：https://nvidia.app.box.com/v/TF170-py35-wTRT
