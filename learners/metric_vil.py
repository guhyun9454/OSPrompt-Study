import numpy as np
import torch
import time
import datetime
from utils.metric import accuracy

def evaluate_task(model, dataloader, device, task_id, class_mask, args):
    """단일 태스크에 대한 평가 함수"""
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            # 2-tuple 또는 3-tuple 모두 처리
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # task_id 인자 제거
            
            # 해당 태스크의 클래스 정확도만 고려
            acc1 = accuracy(outputs, targets, topk=(1,))  # top-1만 사용
            acc1 = float(acc1)  # tensor든 float이든 안전하게 변환
            correct += acc1 * targets.size(0)
            total += targets.size(0)
    
    if total == 0:
        return 0.0
    
    return correct / total

def evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args):
    """
    현재까지의 모든 태스크에 대한 평가 함수
    acc_matrix: 누적 정확도 저장을 위한 행렬 (task_id, task_id)
    """
    # 현재까지의 모든 태스크 평가
    for t in range(task_id + 1):
        acc_matrix[t, task_id] = evaluate_task(
            model, data_loader[t]['val'], device, t, class_mask, args)
    
    # A_i 계산: 각 태스크까지의 평균 정확도
    A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id + 1)]
    A_last = A_i[-1]  # 마지막 태스크까지의 평균 정확도
    A_avg = np.mean(A_i)  # 모든 태스크의 평균 정확도
    
    # Forgetting 계산: 이전 태스크의 최대 성능과 현재 성능의 차이 평균
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
    else:
        forgetting = 0.0
    
    result_str = f"[Average accuracy till task{task_id+1}] A_last: {A_last:.2f} A_avg: {A_avg:.2f}"
    if task_id > 0:
        result_str += f" Forgetting: {forgetting:.4f}"
    print(result_str)
    
    return {"A_last": A_last, "A_avg": A_avg, "Forgetting": forgetting} 