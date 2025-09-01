# Use the official PyTorch image as a base (CPU version)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set the working directory
WORKDIR /workspace

# Copy the RL example script into the container
COPY ./ ./

# (Optional) Install any additional Python packages here
# RUN pip install <other-packages>

# Default command (can be overridden)
CMD ["python", "torch_dist_rl_example.py", "--backend", "nccl"]
