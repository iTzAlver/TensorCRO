FROM tensorflow/tensorflow:2.14.0-gpu
LABEL authors="Alberto Palomo < Eugenio Lorente < Vinicius Garcia"

# Install TensorCRO:
RUN git clone https://github.com/iTzAlver/TensorCRO.git
WORKDIR /TensorCRO
RUN pip install -r ./build/requirements.txt