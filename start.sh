sudo docker run -p 8888:8888 -it --shm-size='2g' --network=host -v ~/.fastai/:/root/.fastai -v ~/Projects/fastai:/workspace/fastai -v ~/Projects/fastcore:/workspace/fastcore -v ~/Projects/timmdocs:/workspace/timmdocs --gpus all -v ~/Projects/run_fastai/:/workspace/run_fastai -w="/workspace" fastdotai/fastai-dev:latest ./run_fastai/run_fastai.sh
