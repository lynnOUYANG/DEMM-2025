# Effective Clustering for Large Multi-Relational Graphs

### Environment Settings

- torch ---- 2.0.1+cu118

- torch-cluster --- 1.6.1+pt20cu118

- torch-geometric ---- 2.6.1

- torch-scatter ---- 2.1.1+pt20cu118

- torch-sparse ---- 0.6.17+pt20cu118

- torch-spline-conv ---- 1.2.2+pt20cu118

- pyg-lib  ---- 0.2.0+pt20cu118

- scipy  ---- 1.13.1

- scikit-learn ---- 1.5.2

- numpy ---- 1.26.3
  
  

### How to run

For DEMM+, you can run the dataset "ACM" by following example:

```
python main.py  --L 5 --alpha 4 --dataset acm-3025 --gamma 0. 
--dim 128  --seed 6  --beta 2.5 --method demm+ --m 10 14  --gpu 0

```

 For DEMM, you can run the dataset "ACM" by following example:

```
python demm-main.py  --alpha 2 --dataset acm-3025 --gamma 0. --seed 6 
--beta 2 --gpu 0

```

For DEMM, you can run the dataset "ACM" by following example:

```
python main.py   --dataset acm-3025 --gamma 0. --dim 6  --seed 6 
 --beta 2 --method demmal --m 10 10 --gpu 0
 
```

For more details about following datasets , you can refer to 'run.sh'.

### Dataset

 Due to space constraints, we have placed all datasets except "oag-cs", "oag-eng", and "rcdd" in the ​**​data.zip​**​ file, which can be accessed via the following link: https://www.dropbox.com/scl/fi/hs9nt4wnjz5l85pxmr7ke/data.zip?rlkey=7k0cefm468zz8vv7j8vrl7ui8&st=o5z53jv6&dl=0. 

For the "oag-cs", "oag-eng", and "rcdd" datasets, their original data access details are provided below:

oag-cs: [graph_CS_20190919.pt - Google 雲端硬碟](https://drive.google.com/file/d/115WygJhRo1DxVLpLzJF-hFCamGc7JY5w/view?usp=drive_link)

oag-eng: [graph_CS_20190919.pt - Google 雲端硬碟](https://drive.google.com/file/d/115WygJhRo1DxVLpLzJF-hFCamGc7JY5w/view?usp=drive_link)

rcdd: https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/''openhgnn/AliRCD_ICDM.zip
