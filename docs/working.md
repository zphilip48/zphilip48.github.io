---
layout: page
title: Working
permalink: /working/
---

<div class="py-2">
    <p class="My project">
    <b>Ongoing My Study on OCR, Facebook Nougat related things</b>
    <p class="Detail">
        1, steup one Face Nougat webserver to trail (done) <br>                    
        <a href="https://www.kaggle.com/code/zphilip/nougat-app">Kaggle webapp Notebook</a><br>
        <a href="https://www.kaggle.com/code/zphilip/nougat-predict">Kaggle manual predict Notebook</a><br>
        <a href="https://gist.github.com/zphilip/91f8f4831470ac530feb38566e9b0892#file-nougat-ipynb">Colab Notebook</a><br>
        2, prepare the training own dataset to have the nougat training from zero (on oning, data is ready) <br>
        a small example data is published on <a href="https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example/data">My Nougat training dataset example</a><br>
        bigger one (ongoing one ) stored https://huggingface.co/datasets/zphilip48/nougat <br>
        3, get LateX-OCR work (done with it's own small training data and notebook) <br>
        4, make the OCR work for both Latex-OCR style data (only one PNG) and also Nougate style data (whole PDF)<br>
            -- modify a little to make the image as inference input , check the notebook  <a href="https://www.kaggle.com/code/zphilip/nougat-app">Kaggle Notebook</a> <br>
            -- <strike> sometime the image inference will just have partially inferenced, unsure why, at same time the pdf version (image conver to pdf) can work well.
                also the image to pdf to image can work well. </strike> <br> 
                <b>fixed by change the bfloat16 to float16 or float32 is good, I guess becauset the P100 (kaggle) don't support bfloat16</b><br>
            -- Nvidia - P40,24G,60K training data with 5 Decoder layer and max_length: 4096 <img src="\images\Snipaste_2023-10-24_14-41-09.png" /> <br>                            
            -- Nvidia-T4 16G,202K training data with 4 decoder layer and max_length: 3584 <img src="\images\Snipaste_2023-10-24_14-44-51.png" /> <br>
    Refer to facebook resources: <br>
    <a href="https://github.com/lukas-blecher/LaTeX-OCR">https://github.com/lukas-blecher/LaTeX-OCR</a><br>
    <a href="https://huggingface.co/facebook/nougat-base">https://huggingface.co/facebook/nougat-base</a><br>
    <a href="https://github.com/facebookresearch/nougat">https://github.com/facebookresearch/nougat</a><br>
        5, programs for AI clients:<br>
        <ul style="list-style-type:disc">
        <li>1) one alpha version desktop program is done (based on pyqt6) , it can snap the desktop or choose the file(pdf) to upload to AI server, I will put download link on the baidu cloud <br> 
        <b>link is :<a href="https://pan.baidu.com/s/1ZkQSfBnpxKTpEot_31HRgA?pwd=1357">https://pan.baidu.com/s/1ZkQSfBnpxKTpEot_31HRgA?pwd=1357</a>  Access code:1357</b><br></li>
        <li>2) one alpha status android app is done for upload the image to AI serser and have the nougat prediction back, it can be downloaded (target is android 12) <a href="/download/app-release.apk">here </a><br></li>                        
        </ul>
    </p> 
</p>   
</div>