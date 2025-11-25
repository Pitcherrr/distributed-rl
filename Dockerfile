FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y build-essential swig && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./ ./

CMD ["python", "ray_nccl_test.py"]
