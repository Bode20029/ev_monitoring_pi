# Start with the NVIDIA L4T PyTorch image
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install additional packages if needed
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgtk2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up locale if not already set
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Copy requirements file
COPY requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# If you need to install additional Python packages, use pip:
# RUN pip install some-package another-package

# Command to run your application
CMD ["python", "your_script.py"]