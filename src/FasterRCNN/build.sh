docker build --no-cache -t anilyerramasu/cxr-detectron2 .

docker run --gpus all --ipc=host --rm -p 5000:5000 -v $(pwd)/input:/opt/output anilyerramasu/cxr-detectron2 
