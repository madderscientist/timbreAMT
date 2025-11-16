# 制作MusicNetEM数据集用于训练
和我的合成数据集不同，musicnet没有单独的音轨，因此只能一股脑输入
要测试在音色无关转录和音色编码上的结果

文件结构
```
├─musicnet/
|   ├─test_data/
|   ├─test_labels/
|   ├─train_data/
|   └─train_labels/
|
├─musicnet_em/ 里面都是mid，是更精准的标签
|
└─musicnet_midis/ 原musicnet的标签，不用
```

## Download
- [origin MusicNet](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)
- [MusicNetEM](https://github.com/benadar293/benadar293.github.io/blob/main/musicnet_em.zip)