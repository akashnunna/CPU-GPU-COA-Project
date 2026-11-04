\# Model Building and Training



\## Project Title

CPU vs GPU Performance Comparison Using Deep Learning



\## 1. Objective

To develop and train a deep learning model using the HAM10000 dataset and study model training on CPU and GPU hardware.



\## 2. Dataset

HAM10000 (Human Against Machine with 10000 training images) contains 10,015 dermatoscopic images belonging to seven classes:



\- akiec

\- bcc

\- bkl

\- df

\- mel

\- nv

\- vasc



The dataset is used for training and validation of the deep learning model.



\## 3. Model Architecture

A Convolutional Neural Network (CNN) is implemented using PyTorch for image classification.



The same model architecture is used for CPU and GPU training to maintain consistency.



\## 4. Implementation Files



| File | Description |

|---|---|

| model.py | CNN architecture |

| dataset.py | Dataset loading and preprocessing |

| train.py | Model training implementation |

| config.py | Configuration settings |



\## 5. Training Experiments



The model was trained using both CPU and GPU environments.



The following artifacts are included:



\- CPU-trained model checkpoint

\- GPU-trained model checkpoint

\- Training and validation accuracy graphs

\- Training and validation loss graphs

\- CPU training history

\- GPU training history



\## 6. Training Graphs



The graphs demonstrate:



\- Training accuracy

\- Validation accuracy

\- Training loss

\- Validation loss



\## 7. Model Checkpoints



\- `models/best\_cpu.pt`

\- `models/best\_gpu.pt`



\## 8. Technologies Used



\- Python

\- PyTorch

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-learn



\## 9. Contribution



Individual responsibility: Model Building and Training.



This contribution covers CNN architecture development, model training using HAM10000, CPU/GPU training experiments, and training performance visualization.



\## 10. Note



This project is an academic CPU vs GPU performance comparison. It is not intended for medical diagnosis.



