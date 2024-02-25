---
layout: page
title:  "working todo"
date:   2023-12-01
categories: working
tags: AI
---

#todo 
- [x] #task compare the nougat small model and code with the nougat big model what's the different between them.  🔼 🛫 2024-01-09 : ✅ 2024-01-21
- transformer version change to  4.34.1, then it work (static saving safe is in newer version)
- [ ] #task based work output of above task let's partially loading the Chinese Bart pre-trained model data 🔼 🛫 2024-01-09 
- [x] prepare one small set dataset include Chinese data for training 🔼 🛫 2024-01-09 ✅ 2024-01-21
- small set dataset --- nougat-dataset-test include - ocrpadded, ocrset
- [ ] adding authentication function to android and pc side client. let's start this after Chinese supporting in server side. 
- [ ] adding authentication and security function to server side. let's start this after Chinese supporting in server side. 
- [ ] today's work 📅 2024-01-11 , ⏫  try to modify the ocr padding method to scale the ocr png to size fit  swin-transformer , then padding.  I suspect the training error is caused by too small png 🛫 padding.  current size is 672(w)x896(h)x8.. arxiv size is 816x1056x24.. something wrong.   
- notice that the orignal code has the resize function to make the small picture to fit the 886*672 size , so using the original un-padding figure (latex ocr dataset image) to feed the current training. but still have the "repetition error"
- put the original un-padding figure  and the padded figure together into the training , it seems that the training have no "repetition error" still unclear why it is so????
	- this have the repetition error... while using bigger dataset from arXiv orignal data
	- I am think the VIT how to train the all blank image... it is lead to some error , or how it is treated , get some test ???
	- how about using all back or white image as training data 
	- how the voice to text treat the white noise? how the blank image is treated 
	- in normal ViT , the fixed size image is always required...,   how about reserve the SWIN transformer architecture, change from low level resolution to high resolution 
- [x] bleu score ✅ 2024-01-12, the bleu score is noted at [[Transformer_learning#3.2 Bleu Score]]
- [ ] Pytorch-view的用法
- 在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize（）的功能，但是用法可能不太一样。如下例所示

```text
>>> import torch
>>> tt1=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
>>> result=tt1.view(3,2)
>>> result
tensor([[-0.3623, -0.6115],
        [ 0.7283,  0.4699],
        [ 2.3261,  0.1599]])
```

1. torch.view(参数a，参数b，...)

在上面例子中参数a=3和参数b=2决定了将一维的tt1重构成3x2维的张量。

2. 有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况。
```text
>>> import torch
>>> tt2=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt2.view(-1)
>>> result
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
```
由上面的案例可以看到，如果是torch.view(-1)，则原张量会变成一维的结构。
```text
>>> import torch
>>> tt3=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt3.view(2,-1)
>>> result
tensor([[-0.3623, -0.6115,  0.7283],
        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。

