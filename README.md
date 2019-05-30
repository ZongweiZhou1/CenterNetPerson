
##Pedestrian detection based on CenterNet

In this repo, we re-train the centernet on CityPerson dataset to get a pedestrian detector
[CenterNet](https://github.com/Duankaiwen/CenterNet)


##Preparation

Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CenterNet
```

## Compiling Corner Pooling Layers
```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS
```
cd <CenterNet dir>/external
make
```

## CityPerson dataset

- Download the CityPerson dataset and label files in [images](https://www.cityscapes-dataset.com/file-handling/?packageID=3), [label](https://www.cityscapes-dataset.com/file-handling/?packageID=28)
- create a softlink in `data` to your CityPerson data
    ```
    ln -s  #to/yourdata/CityPerson data/
    ```
 
## Training and Evaluation
To train CenterNet-52
```buildoutcfg
python train.py --cfg_file CenterNet-52
```
The default configure in `config/CenterNet-52.json` is 2 (12G) GPUs and batchsize=12, you can modify them according to your case.

To evaluate your detector
```buildoutcfg
python test.py --cfg_file CenterNet-52 --testiter  #checkpoint_epoch
```

## Demo
The demo images are stored in `data/demo`
```buildoutcfg
python demo.py
```