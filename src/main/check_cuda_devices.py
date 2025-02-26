import torch

def list_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        return devices
    else:
        return "CUDA is not available."

if __name__ == "__main__":
    devices = list_cuda_devices()
    print("Available CUDA devices:", devices)
