[YOLOv8](https://github.com/ultralytics/ultralytics) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.<br>

Horizon Algorithm Toolchain (also called OpenExplorer) is self-developed by Horizon Robotics to enable efficient and accurate model deployment onto the Horizon SoCs. It is released in the form of development package which can be obtained via [Horizon Algorithm Toolchain XJ3](https://developer.horizon.cc/forumDetail/136488103547258769) and [Horizon Algorithm Toolchain J5](https://developer.horizon.cc/forumDetail/118363912788935318).<br>

This project aims to deploy YOLOv8 models onto the Sunrise-3 SoCs of Horizon Robotics efficiently through Horizon Algorithm Toolchain. The deployment pipeline involves floating-point model training and exporting to ONNX format, model quantization and optimization, model compilation and model deployment (only model performance evaluation on board showed).<br>

## <div align="center">Quick Start</div>

<details open>
<summary>Install</summary>

#### YOLOv8
`yolov8_x3/ultralytics` folder contains the official YOLOv8 development tools along with the modified model structure that adapts to the Horizon platforms. For detailed model modification explanations, please see [【前沿算法】地平线适配 YOLOv8](https://developer.horizon.cc/forumDetail/189779523032809473). Run the following commands to install:
```bash
cd yolov8_x3/ultralytics
pip install -r requirements.txt
python setup.py install
```

#### Horizon Algorithm Toolchain
Horizon Algorithm Toolchain provides Docker images to provide users with quick access to the tools. Simply run the `run_docker.sh` script:
```bash
sh run_docker.sh ./data/ gpu
pip uninstall horizon-nn
pip install ddk/package/host/ai_toolchain/horizon_nn_gpu-0.18.2-cp38-cp38-linux_x86_64.whl
```
This will specify the dataset path and run the docker in GPU mode (make sure that the GPU environment is properly set).

</details>

<details open>
<summary>Usage</summary>

#### Build
This will enable Horizon Algorithm Toolchain to automatically convert the ONNX model to formats ready for deployment on Horizon SoCs.
```bash
sh build.sh
```

#### Evaluate on Host-side
Before running the following commands, please download [coco_val2017](https://cocodataset.org/) under folder `yolov8_x3/ptq_project/coco` and specify current dataset path in file `evaluate.sh`.

To evaluate the accuracy of original ONNX model with full coco_val2017:
```bash
sh evaluate.sh origin
```

To evaluate the accuracy of quantized model with full coco_val2017:
```bash
sh evaluate.sh quanti
```

To evaluate the accuracy of original ONNX model just for test:
```bash
sh evaluate.sh origin 20
```

To evaluate the accuracy of quantized model just for test:
```bash
sh evaluate.sh quanti 20
```

</details>


