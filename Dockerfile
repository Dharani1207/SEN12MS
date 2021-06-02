FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# install dependencies 
RUN conda install -c conda-forge cupy 
RUN conda install -c conda-forge opencv
RUN pip install scipy sklearn rasterio natsort matplotlib scikit-image pandas tqdm natsort tensorboardX

RUN mkdir -p ./classification
RUN mkdir -p ./labels
RUN mkdir -p ./label_split_dir
RUN mkdir -p ./splits
RUN mkdir -p ./utils

# add directories
ADD classification ./classification 
ADD labels ./labels
ADD label_split_dir ./label_split_dir
ADD splits ./splits
ADD utils ./utils

