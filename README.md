# Keras-ocr-cpp
Keras OCR implemented in C++ based on [faustomorales/keras-ocr](https://github.com/faustomorales/keras-ocr) repo.
This is a small side project for my research, where python couldn't be used. Thus, it may lack some of the features available in original repo.

---

## Getting started

### Requirements
In this project two libraries were used:
- [cppflow](https://github.com/serizba/cppflow) - C++ wrapper for tensor manipulation and eager execution. Installation is described in repo.
- [Tensorflow C API](https://www.tensorflow.org/install/lang_c) - use their documentation or description in [cppflow docs](https://serizba.github.io/cppflow/installation.html) to install TF C API locally/globally. For Apple Silicon CPUs compilation from sources is required, because the version provided in docs is only for x86.

### Models
After installing the dependencies, export the models to protobuf format (model + weights). This can be done using python in keras-ocr.
It's best to use [Poetry](https://python-poetry.org/) - python dependency manager. Follow the documentation to install poetry.

Install all requirements for keras-ocr, activate created venv and run python.

```bash
poetry install
poetry shell
python
```

Then, execute following code, which will export models to specified directories.

```python
import keras_ocr


pipeline = keras_ocr.pipeline.Pipeline()
pipeline.detector.model.save('/path/to/models/craft-detection')
pipeline.recognizer.prediction_model.save('/path/to/models/crnn-recognition')
```

Export or move them to [models](models) directory or modify [CMakeLists.txt](keras_ocr/CMakeLists.txt).

### Usage
```bash
mkdir build
cd build
cmake ../keras_ocr
cmake --build .
./KerasOCR
```

Example images were downloaded from Wikipedia. Sources:
- img1.jpg: https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg
- img2.jpg: https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg

## Advanced configuration
The execution of inference is entirely on the side of cppflow. If you need to make changes, install the library locally and make modifications to it.