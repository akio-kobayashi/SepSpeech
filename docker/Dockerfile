# 必要な場合はproxy設定を追加する
# nvidia-driver >= 515 && pytorch-2.0 binary-build
ARG CUDA="11.8.0"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
# rootで実行
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends emacs python3.9 python3-pip libsndfile1-dev cmake
# unfortunatelly, we do not use CUDA 12.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip3 install torchmetrics lightning einops numpy sklearn tqdm pandas scipy numba h5py seaborn wave librosa transformers python-box
RUN pip3 install tensorboard
#COPY warp-transducer.tar /
#RUN tar xvf /warp-transducer.tar
#RUN mkdir /warp-transducer/build
#RUN cd /warp-transducer/build
#RUN cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local /warp-transducer
#RUN make
#RUN make install
#RUN cd /warp-transducer/pytorch_binding
#RUN export CUDA_HOME="/usr/local/cuda"
#RUN python3 /warp-transducer/pytorch_binding/setup.py install
#RUN cp -f /warp-transducer/build/libwarprnnt.so /usr/local/lib/