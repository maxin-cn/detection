import torch
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import SlimPruner, L1NormPruner, FPGMPruner
from nni.compression.pytorch.utils import not_safe_to_prune

device = 'cuda:0'
# device = 'cpu'
# config = 'configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py'
config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
checkpoint = None
img_file = 'demo/demo.JPEG'

# build the model from a config file and a checkpoint file
model = init_detector(config, checkpoint, device=device)

# model.forward = model.forward_dummy_combined
# model.forward = model.forward_dummy
model.forward = model.forward_dummy_combined

# pre_flops, pre_params, _ = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
# inputs_combined = torch.randn(1, 3, 256, 256).to(device)
inputs_combined = torch.randn(1, 4, 256, 256).to(device)
# im = torch.ones(1, 3, 256, 256).to(device)
# proposals = torch.randn(1000, 4).to(device)
# inputs_combined = (im, proposals)
# out = model(im, proposals)
out = model(inputs_combined)
# torch.jit.trace(model.forward_dummy_combined, (), strict=False)
# torch.jit.trace(model, (im, proposals))
torch.jit.trace(model, inputs_combined)


# with torch.no_grad():
#     input_name = ['input']
#     output_name  = ['output']
#     onnxname = 'fanet.onnx'
#     torch.onnx.export(model, im, onnxname, input_names = input_name, output_names = output_name,
#                     opset_version=11, training=False, verbose=False, do_constant_folding=False)
#     print(f'successful export onnx {onnxname}')
# exit()

# scores = model(return_loss=False, **data)
# scores = model(return_loss=False, **im)

# test a single image
# result = inference_model(model, img_file)

# Start to prune and speedupls
print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
# not_safe = not_safe_to_prune(model, (im, proposals))
not_safe = not_safe_to_prune(model, inputs_combined)


print('\n' + '=' * 50 +  'not_safe' + '=' * 50, not_safe)
cfg_list = []
for name, module in model.named_modules():
    print(name)
    if name in not_safe:
        continue
    if isinstance(module, torch.nn.Conv2d):
        cfg_list.append({'op_types':['Conv2d'], 'sparsity':0.2, 'op_names':[name]})

print('cfg_list')
for i in cfg_list:
    print(i)

pruner = FPGMPruner(model, cfg_list)
_, masks = pruner.compress()
pruner.show_pruned_weights()
pruner._unwrap_model()
pruner.show_pruned_weights()


# ModelSpeedup(model, dummy_input=(im, proposals), masks_file=masks, confidence=32).speedup_model()
ModelSpeedup(model, dummy_input=inputs_combined, masks_file=masks, confidence=16).speedup_model()
# torch.jit.trace(model, (im, proposals), strict=False)
torch.jit.trace(model, inputs_combined, strict=False)
print(model)
exit()
# flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
# print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M')
# print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M')
model.forward = model.forward_
torch.save(model, 'chek/prune_model/res2net50-w14-s8_8xb32_cifar10_sparsity_0.2.pth')