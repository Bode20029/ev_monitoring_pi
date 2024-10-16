# import torch
# print("CUDA Available: ", torch.cuda.is_available())
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

# import torch

# def check_cuda_availability():
#     if torch.cuda.is_available():
#         print("CUDA is available.")
#         print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
#     else:
#         print("CUDA is not available.")

# if __name__ == "__main__":
#     check_cuda_availability()

#################
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")

######################
# import cv2
# print(f"OpenCV version: {cv2.__version__}")
# print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
# if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#     print(f"OpenCV CUDA version: {cv2.cuda.getDevice()}")

# import numpy
# print(numpy.__version__)

import torch
import sys

def test_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Try a simple CUDA operation
        x = torch.rand(5, 3)
        print(f"Input tensor:\n{x}")
        if torch.cuda.is_available():
            x = x.cuda()
            print("Tensor successfully moved to CUDA")
        print(f"Tensor device: {x.device}")
        
        # Perform a simple operation
        y = x * 2
        print(f"Output tensor (x * 2):\n{y}")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    test_cuda()