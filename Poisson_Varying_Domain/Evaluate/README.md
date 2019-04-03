# Evaluating Trained Models

## Freezing Models
Trained models can be frozen using the utility file `Freeze.py`:

```console
$ python Freeze.py --model_dir ../Model/
```


## Plotting Predictions

```console
$ python Plot_Prediction.py --model_dir ../Model/
```

<p align="center">
  <img width="700" src="../../figures/Prediction_Plot.png" style="margin: auto;">
</p>


```console
$ python Plot_Prediction.py --model_dir ../Model/ --show_error
```

<p align="center">
  <img width="700" src="../../figures/Error_Plot.png" style="margin: auto;">
</p>



## Uncertainty Quantification Analysis

```console
$ python Compute_UQ_Bounds.py --model_dir ../Model/

$ python Plot_UQ.py
```

<p align="center">
  <img width="700" src="../../figures/Plot_UQ.png" style="margin: auto;">
</p>



## Uncertainty Levels during Training

```console
$ python Plot_Training_UQ.py --model_dir ../Model/
```

<p align="center">
  <img width="500" src="../../figures/Plot_Training_UQ_1.png" style="margin: auto;">
</p>

<p align="center">
  <img width="500" src="../../figures/Plot_Training_UQ_2.png" style="margin: auto;">
</p>


## Class Losses

```console
$ python Compute_Class_Losses.py --model_dir ../Model/

$ python Plot_Class_Loss.py
```

<p align="center">
  <img width="700" src="../../figures/Plot_Class_Loss.png" style="margin: auto;">
</p>



## Analysis of Data Counts

```console
$ python main.py --model_dir Model_1-4  --data_files 4
$ python main.py --model_dir Model_1-8  --data_files 8
$ python main.py --model_dir Model_1-16 --data_files 16
$ python main.py --model_dir Model_1-32 --data_files 32
```


```console
$ python Plot_Data_Count_Comparisons.py
```

<p align="center">
  <img width="700" src="../../figures/Plot_Data_Count_Comparisons.png" style="margin: auto;">
</p>
