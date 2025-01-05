# Face Recognition Project

This is the upgraded model of face recognition model built with TensorFlow, Keras, and OpenCV. It provides the ability to train a model on a custom dataset, save it, and later use it for real-time face recognition using a webcam. It  is easy and most accurate model.

## Prerequisites

Before you start, make sure you have the following installed:

- Python 3.7 or later
- TensorFlow 2.x
- OpenCV
- Other dependencies listed in `requirements.txt`

## Dataset Structure

The dataset should be structured as follows: 

dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... `use atlest 50 images of each person.`
└── Person2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...`use atlest 50 images of each person .`

## Installation

Follow these steps to set up the environment and run the project:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Arya182-ui/upgraded_face_recognition_model.git
cd Face-Recognition
```
### 2.Install requirements
```bash

pip install -r requirements.txt


```

### 3. Prepare data set 

``` bash 
python takeImage.py
```


### 4.Train the model 

``` bash 

python train_model.py


```

### 5.Use the model 

```bash 

python main.py

```


### **Contributing**

We welcome contributions to improve this project! Whether it's bug fixes, new features, improvements to documentation, or other enhancements, your contributions are appreciated. Here’s how you can contribute:

#### How to Contribute

1. **Fork the Repository**  
   Click on the "Fork" button on the top-right of the repository page to create your own copy of the repository.

2. **Clone the Forked Repository**  
   Clone your forked repository to your local machine using the following command:

   ```bash
   https://github.com/Arya182-ui/upgraded_Face_Recognition_Model.git
   ``` 
  


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Author

[Ayush Gangwar](https://github.com/Arya182-ui)