import torch


print(torch.cuda.is_available())

print(torch.cuda.device_count())
print(torch.cuda.current_device())

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_device_capability(0))

print(torch.cuda.mem_get_info(0))