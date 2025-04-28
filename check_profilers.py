import os
import argparse
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmengine.analysis import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
from calflops import calculate_flops
from deepspeed.profiling.flops_profiler import get_model_profile
from torch.profiler import profile, ProfilerActivity


class SimpleLinearLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(100, 100)
    
    def forward(self, x):
        return self.linear(x)


class LinearBatchNorm(torch.nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


class SharedLinearLayers(torch.nn.Module):
    def __init__(self, shared: bool, **kwargs):
        super().__init__(**kwargs)

        self.linears = nn.ModuleList()

        for _ in range(3):
            self.linears.append(
                LinearBatchNorm(100, 100)
            )
        
        if shared:
            for i in range(3):
                self.linears[i].linear = self.linears[0].linear

    def forward(self, x):
        x1 = x[:, :100]
        x2 = x[:, 100:200]
        x3 = x[:, 200:]
        x = (x1, x2, x3)

        results = []
        for idx, input_feat in enumerate(x):
            output_feat = self.linears[idx](input_feat)
            results.append(output_feat)
        
        return tuple(results)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["linear", "rtmdet", "unshared_linears", "shared_linears"])
    parser.add_argument("--profilers", nargs="+", type=str, default=["mmengine", "fvcore", "calflops", "deepspeed", "torch_profiler"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # prepare model and input shape
    if args.model_type == "linear":
        model = SimpleLinearLayer()
        input_shape = (1, 100)
    elif args.model_type == "rtmdet":
        cfg = Config.fromfile("/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py")
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        model = MODELS.build(cfg.model)
        input_shape = (1, 3, 640, 640)
    elif args.model_type == "unshared_linears":
        model = SharedLinearLayers(shared=False)
        input_shape = (1, 300)
    elif args.model_type == "shared_linears":
        model = SharedLinearLayers(shared=True)
        input_shape = (1, 300)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()


    # print
    print_str = f"Profiling results on {args.model_type} with {', '.join(args.profilers)}\n"
    if args.model_type in ["linear", "unshared_linears", "shared_linears"]:
        div = 1e3
        unit = "KFLOPs"
    elif args.model_type == "rtmdet":
        div = 1e9
        unit = "GFLOPs"
    else:
        raise NotImplementedError

    # mmengine
    if "mmengine" in args.profilers:
        outputs = get_model_complexity_info(
            model,
            input_shape=tuple(input_shape[1:]),
            inputs=None,
            show_table=False,
            show_arch=True)
        flops = outputs['flops']
        with open(os.path.join(args.output_dir, f'{args.model_type}_mmengine.yaml'),'w') as f:
            print(outputs['out_arch'], file=f)
        print_str += f"mmengine: {int(flops / div)} {unit}, (MACs actually)\n"

    # fvcore
    if "fvcore" in args.profilers:
        flops = FlopCountAnalysis(model, torch.zeros(input_shape).to("cuda"))
        flops = flops.total()
        print_str += f"fvcore: {int(flops / div)} {unit}, (MACs actually)\n"

    # calflops
    if "calflops" in args.profilers:
        flops, _, _ = calculate_flops(model=model, 
                                            input_shape=input_shape,
                                            print_results=False,
                                            output_as_string=False,
                                            output_precision=4)
        print_str += f"calflops: {int(flops / div)} {unit}\n"

    # deepspeed
    if "deepspeed" in args.profilers:
        flops, _, _ = get_model_profile(model=model, # model
                                        input_shape=input_shape, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=10, # the number of warm-ups before measuring the time of each module
                                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=os.path.join(args.output_dir, f'{args.model_type}_deepspeed.yaml'), # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
        print_str += f"deepspeed: {int(flops / div)} {unit}\n"

    # pytorch profiler
    if "torch_profiler" in args.profilers:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            model(torch.zeros(input_shape).to("cuda"))
        flops = sum(event.flops for event in prof.key_averages())
        print_str += f"torch profiler: {int(flops / div)} {unit}"


    print("\n")
    print(print_str)
    print("\n")
