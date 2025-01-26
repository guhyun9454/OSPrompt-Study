import torch
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis
import timm


# 모델 분석 함수
def analyze_model(model, input_shape=(1, 3, 224, 224), num_iterations=100):
    """
    주어진 모델의 FLOPs, 파라미터 수, 출력 텐서 크기를 분석하고,
    추론을 100번 반복하여 평균 실행 시간을 출력
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.rand(input_shape).cuda()  
    model = model.cuda()  

    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total() / 1e9  # GFLOPs 단위로 변환

    total_params = count_parameters(model)

    # 추론 시간 측정 (100회 반복)
    model.eval()
    with torch.no_grad():
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()  # GPU 동기화
            end_time = time.time()
            times.append(end_time - start_time)

    # 평균 실행 시간 계산
    avg_inference_time = sum(times) / len(times)

    # 모델 추론 결과 확인
    output_shape = model(dummy_input).shape

    # 결과 출력
    print(f"Model: {model.__class__.__name__}")
    print(f"FLOPs: {total_flops:.2f} GFLOPs")
    print(f"Parameters: {total_params:,}")
    print(f"Output Shape: {output_shape}")
    print(f"Average Inference Time (over {num_iterations} runs): {avg_inference_time * 1000:.2f} ms")

# 예제 실행
if __name__ == "__main__":
    
    print("ViT")
    model = timm.create_model('vit_base_patch16_224',pretrained=True)
    analyze_model(model)

    print("ResNet50")
    zoo_model_query = timm.create_model('resnet50', pretrained=True)
    zoo_model_query = nn.Sequential(*list(zoo_model_query.children())[:-1])  # Feature extraction laye
    analyze_model(zoo_model_query)

    print("ResNet101")
    zoo_model_query = timm.create_model('resnet101', pretrained=True)
    zoo_model_query = nn.Sequential(
        *list(zoo_model_query.children())[:-1],
        nn.Unflatten(1, (1, -1)),
        nn.AdaptiveAvgPool1d(768),
        nn.Flatten(1)
    )
    analyze_model(zoo_model_query)

    print("ResNet152")
    zoo_model_query = timm.create_model('resnet152', pretrained=True)
    zoo_model_query = nn.Sequential(
        *list(zoo_model_query.children())[:-1],
        nn.Unflatten(1, (1, -1)),
        nn.AdaptiveAvgPool1d(768),
        nn.Flatten(1)
    )
    analyze_model(zoo_model_query)

    print("ConvNeXt Small")
    zoo_model_query = timm.create_model('convnext_small', pretrained=True)
    zoo_model_query = nn.Sequential(
        *list(zoo_model_query.children())[:-1],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    analyze_model(zoo_model_query)

    print("ConvNeXt Tiny")
    zoo_model_query = timm.create_model('convnext_tiny', pretrained=True)
    zoo_model_query = nn.Sequential(
        *list(zoo_model_query.children())[:-1],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    analyze_model(zoo_model_query)

    print("ConvNeXt Base")
    zoo_model_query = timm.create_model('convnext_base', pretrained=True)
    zoo_model_query = nn.Sequential(
        *list(zoo_model_query.children())[:-1],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Unflatten(1, (1, -1)),
        nn.AdaptiveAvgPool1d(768),
        nn.Flatten(1)
    )
    analyze_model(zoo_model_query)
