# 0. Docker 환경 준비
```bash
git clone https://github.com/richardSHkim/profiling
cd profiling/docker
docker build -t profiling .
```
# 1. MMENGINE의 `get_model_complexity_info` 함수는 FLOPs가 아닌 MACs 값을 보고함.
### Single Linear Layer (100x100) 에 대한 FLOPs 계산.
```bash
docker run --gpus all --rm profiling linear
```
- 실제 $N \times M$ fully connected layer의 MACs와 FLOPs는 각각 $N \times M$ 과 $2 \times N \times M$ 이므로, 20 KFLOPs가 나와야 합니다.
- 하지만, mmengine의 `get_model_complexity_info` 함수는 FLOPs가 아닌 MACs 값인 10k 를 보고하고 있습니다.
### RTMDET-Ins-s 에 대한 FLOPs 계산.
```bash
docker run --gpus all --rm profiling rtmdet
```
- input shape [1x3x640x640] 기준입니다.
- [RTMDET 논문](https://arxiv.org/pdf/2212.07784) Table 3. 에서 RTMDET-Ins-s 의 FLOPs 값을 21.5 로 보고하고 있습니다.
- [RTMDET official code](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet#instance-segmentation) 에서도 동일하게 RTMDET-Ins-s 의 FLOPs 값을 21.5 로 보고하고 있습니다.
- 실제 위 코드를 실행하면, mmengine은 21 GFLOPs 를 보고합니다.
- 하지만, 이는 위에서 확인한 것과 같이 MACs 수치입니다.
- RTMDET 논문의 Table 3. 에서 비교군으로 사용된 YOLOv5 계열의 FLOPs 역시 [YOLOv5 official repository](https://github.com/ultralytics/yolov5?tab=readme-ov-file#%EF%B8%8F-segmentation) 에서 보고하는 FLOPs 값의 절반인 MACs 값을 보고하고 있습니다.
### (참고) MMENGINE `Unsupported operator` Warning.
```bash
mmengine - WARNING - Unsupported operator aten::silu_ encountered 92 time(s)
```
- `rtmdet` 옵션으로 코드를 실행하면 위와 같은 경고 메시지가 다수 발생하지만, 실제 해당 operation 들은 전체 GFLOPs에 영향을 주지 않을 만큼 작은 값입니다.


# 2. DeepSpeed Profiler 오류 구현
### 과도한 MMENGINE FLOPs 값과 DeepSpeed FLOPs 값 차이
- mmengine에서 보고한 RTMDET-Ins-s 에 대한 FLOPs 값이 MACs 값이라면, 실제 FLOPs 값은 21G의 약 2배인 42 GFLOPs가 되어야 합니다.
- 하지만, DeepSpeed에서는 21G의 4배 이상의 값인 92 GFLOPs를 보고하고 있습니다.
- mmengine과 deepspeed profiling 결과를 1:1 로 비교해본 결과, deepspeed 쪽에서 RTMDET의 bbox_head 특히, shared detection head인 `self.cls_convs`와 `self.reg_convs` 쪽에서 FLOPs 값이 중복으로 더해지고 있는 것을 발견하였습니다.
### Shared Linears
![shared linears architecture](asset/shared_linears.png)
- 중복 계산 오류를 검증하기 위해 위 그림과 같이 fully connected layer 3개와 batch norm으로 구성된 네트워크를 준비하였습니다. (그림에서 batch norm은 생략되었습니다.)
- 100x100 fully connected layer는 10 KMACs 연산량을 가지고 있습니다.
- [1, 300] 크기의 tensor가 [1, 100] 크기의 tensor 3개로 쪼개져서 각각 fully connected layer의 input으로 들어갑니다.
- 따라서, 모델은 총 30 KMACs (60 KFLOPs)의 연산량을 가지고 있습니다.
- `shared` 옵션을 `True`로 설정하면 parameter 수를 1/3로 절약할 수 있지만, FLOPs 관점에서는 차이가 없어야 합니다.

### 오류 구현
```bash
docker run --gpus all --rm profiling unshared_linears
docker run --gpus all --rm profiling shared_linears
```
- 해당 모델에 대해서 `shared` 옵션을 `False`로 했을 때와 `True`로 했을 때의 FLOPs 값을 비교해보면, mmengine과 torch profiler는 값의 차이가 없습니다.
- 반면, calflops와 deepspeed의 경우 `shared=True`로 설정했을 때, FLOPs가 중복으로 계산되어 60 KFLOPs의 3배인 180 KFLOPs를 보고합니다.
- 해당 오류는 deepspeed에 (issue)[#TODO: 링크]로 남겨놓았습니다.

### PyTorch Profiler 결과
- 위 FLOPs/MACs 테스트와 shared linears 테스트에서 항상 정확한 결과를 보고한 profiler는 pytorch profiler가 유일했습니다.
- pytorch profiler는 RTMDET-Ins-s 모델에 대해서 43 GFLOPs 값을 보고합니다.

# 3. Leaderboard Score 수정
### RTMDET-Ins-s 모델의 FLOPs
- mmengine, fvcore의 결과는 사실 MACs 값이었고, calflops, deepspeed는 shared layer에 대해서 중복된 계산값을 보고하고 있었습니다.
- 저희는 RTMDET-Ins-s 모델의 FLOPs 값으로 pytorch profiler의 계산 값인 43 GFLOPs를 사용하고자 합니다.

### Leaderboard Score
$0.1 \times exp(-21 \div 136) - 0.1 \times exp(-43 \div 136) = 0.012261$
- Score 계산식에 21 GFLOPs 대신 43 GFLOPs를 대입하면 0.012261 점이 깎이게 됩니다.
- 저희가 제출한 최종 score 0.50571 점에서 해당 값을 빼면, 0.49345 점이 됩니다.