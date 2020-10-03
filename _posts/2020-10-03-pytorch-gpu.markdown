---
layout: post
title:  "Pytorch에서 GPU 사용하기"
date:   2020-10-3 00:00:00
author: y-rok
categories: 
- pytorch
tags: 
- pytorch
- gpu
---

이번 포스트에서는 **Pytorch**에서 **GPU**를 활용하는 방법에 대해 다음 3가지로 소개합니다. ([공식 문서](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics)를 참고하여 작성.) 

- 어떻게 gpu를 사용하는가?
- argument에 따라 cpu 혹은 gpu에서 동작하도록 코드 작성
- 특정 gpu 사용하기    

## 어떻게 gpu를 사용하는가?

tensorflow에서는 1.15 이후의 버전부터는 gpu에 자동으로 tensor들이 할당되지만 **<u>pytorch에서는 gpu에 해당 tensor를 올리라고 코드를 작성해주어야 합니다.</u>** 다음의 코드를 통해 gpu에 tensor를 올려봅시다.

```python

# tensor를 gpu에 할당하는 3가지 방법 ("cuda" -> default cuda device (default gpu device))
x = torch.tensor([1., 2.], device="cuda") 
x = torch.tensor([1., 2.]).cuda() 
x = torch.tensor([1., 2.]).to("cuda") 

print(x.device) # 할당된 device 정보 -> cuda:0
```

## argument에 따라 cpu 혹은 gpu에서 동작하도록 코드 작성

**<u>하지만, 위의 코드처럼 작성시 gpu가 없는 경우 error가 발생할 수 있습니다.</u>** 따라서, 실제로 코딩 시에는 다음과 같이 작성하여 arugment에 따라 cpu 혹은 gpu를 사용하도록 합시다.

```python
"""
	python main.py --cpu -> cpu에서 동작
	python main.py -> gpu에서 동작
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',help='run in cpu') 
args = parser.parse_args()

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    
 x = torch.tensor([1., 2.]).to(device) # 설정된 device에 tensor 할당 
```

## 특정 gpu 사용하기

gpu가 여러대 있을 경우 특정 gpu를 사용해야할 때가 있습니다. 이를 위한 설정 방법은 2가지가 있습니다.

- shell에서 환경 변수로 설정하기

```bash
# GPU 1을 사용하기
>> export CUDA_VISIBLE_DEVICES=1
>> python main.py
```

- Code 내에서 작성하기

```python
device = torch.device('cuda:1')
x = torch.tensor([1., 2.]).to(device) # GPU 1에 할당

# 이 안에서는 기본적으로 GPU 0에 할당
with torch.cuda.device(0):
    # GPU 0 에 할당 
    x = torch.tensor([1., 2.]).to("cuda")
    
    # 단, 여기서도 다른 gpu에 할당 가능
    x = torch.tensor([1., 2.]).to(device) # GPU 1에 할당
    
```

