<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Boston House Price Prediction
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-bikesharingcountregression.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1NQnRjmpo-ZsPGS2nGT783lb_0LdUWHpm)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Pre-processing__](#c-pre-processing)
    - [__(D) Exploratory Data Analysis__](#d-exploratory-data-analysis)
    - [__(E) Modelling__](#e-modelling)
    - [__(F) Saving the project__](#f-saving-the-project)
    - [__(G) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __regression__ project that uses the  [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to __predict the rented bike counts__.
- The __goal__ is build a model that accurately __predict the rented bike counts__  based on the features. 
- The performance of the model is evaluated using several __metrics__, including _MaxError_, _MeanAbsoluteError_, _MeanAbsolutePercentageError_, _MSE_, _RMSE_, _MAE_, _R2_, _ExplainedVariance_ and other imbalanced regression metrics.

#### __Overview__
- This project involves building a machine learning model to predict the purchase amounts based on number of 17 features. 8 features are categorical and 9 features are numerical. The dataset contains 17379 records. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, Scikit-learn.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-bikesharingcountregression.hf.space" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1jCJUsewX5vzefILly8yRF6OMSDbxlp83
"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/BikeSharingCountRegression/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1NQnRjmpo-ZsPGS2nGT783lb_0LdUWHpm"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __predict the rented bike counts__  based on features.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-bikesharingcountregression.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-bikesharingcountregression.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) from kaggle platform.
- The dataset contains 17 features, 8 features are categorical and 9 features are numerical.
- The dataset contains the following features:


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>



| Attribute | Description |
| --- | --- |
| instant | Record index |
| dteday | Date |
| season | Season (1:winter, 2:spring, 3:summer, 4:fall) |
| yr | Year (0: 2011, 1:2012) |
| mnth | Month (1 to 12) |
| hr | Hour (0 to 23) |
| holiday | Weather day is holiday or not (extracted from [Web Link]) |
| weekday | Day of the week |
| workingday | If day is neither weekend nor holiday is 1, otherwise is 0 |
| weathersit | <ul><li>1: Clear, Few clouds, Partly cloudy, Partly cloudy</li><li>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist</li><li>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds</li><li>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog</li></ul> |
| temp | Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale) |
| atemp | Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale) |
| hum | Normalized humidity. The values are divided to 100 (max) |
| windspeed | Normalized wind speed. The values are divided to 67 (max) |
| casual | Count of casual users |
| registered | Count of registered users |
| cnt | Count of total rental bikes including both casual and registered |



</td></tr> </table>


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

| #   | Column      | Non-Null Count | Dtype    |
| --- | ----------- | -------------- | -------- |
| 0   | instant     | 17379 non-null | int64    |
| 1   | dteday      | 17379 non-null | object   |
| 2   | season      | 17379 non-null | int64    |
| 3   | yr          | 17379 non-null | int64    |
| 4   | mnth        | 17379 non-null | int64    |
| 5   | hr          | 17379 non-null | int64    |
| 6   | holiday     | 17379 non-null | int64    |
| 7   | weekday     | 17379 non-null | int64    |
| 8   | workingday  | 17379 non-null | int64    |
| 9   | weathersit  | 17379 non-null | int64    |
| 10  | temp        | 17379 non-null | float64  |
| 11  | atemp       | 17379 non-null | float64  |
| 12  | hum         | 17379 non-null | float64  |
| 13  | windspeed   | 17379 non-null | float64  |
| 14  | casual      | 17379 non-null | int64    |
| 15  | registered  | 17379 non-null | int64    |
| 16  | cnt         | 17379 non-null | int64    |



</td><td>

<div style="flex: 50%; padding-left: 50px;">

|      |    count |        mean |         std |   min |      25% |      50% |      75% |      max |
|:-----|---------:|------------:|------------:|------:|---------:|---------:|---------:|---------:|
|instant     | 17379 |  8690.00000 | 5017.029500 | 1.00 | 4345.5000 | 8690.0000 | 13034.5000 | 17379.0000 |
|season      | 17379 |     2.50164 |    1.106918 | 1.00 |    2.0000 |    3.0000 |    3.0000 |    4.0000 |
|yr          | 17379 |     0.50256 |    0.500008 | 0.00 |    0.0000 |    1.0000 |    1.0000 |    1.0000 |
|mnth        | 17379 |     6.53778 |    3.438776 | 1.00 |    4.0000 |    7.0000 |   10.0000 |   12.0000 |
|hr          | 17379 |    11.54675 |    6.914405 | 0.00 |    6.0000 |   12.0000 |   18.0000 |   23.0000 |
|holiday     | 17379 |     0.02877 |    0.167165 | 0.00 |    0.0000 |    0.0000 |     0.0000 |     1.0000 |
|weekday     | 17379 |     3.00368 |    2.005771 | 0.00 |    1.0000 |    3.0000 |     5.0000 |     6.0000 |
|workingday  | 17379 |     0.68272 |    0.465431 | 0.00 |    0.0000 |    1.0000 |     1.0000 |     1.0000 |
|weathersit  | 17379 |     1.42528 |    0.639357 | 1.00 |    1.0000 |    1.0000 |     2.0000 |     4.0000 |
|temp        | 17379 |     0.49699 |    0.192556 | 0.02 |    0.3400 |    0.5000 |     0.6600 |     1.0000 |
|atemp       | 17379 |     0.47578 |    0.171850 | 0.00 |    0.3333 |    0.4848 |     0.6212 |     1.0000 |
|hum         | 17379 |     0.62723 |    0.192930 | 0.00 |    0.4800 |    0.6300 |     0.7800 |     1.0000 |
| windspeed | 17379.0 | 0.190098   | 0.122340    | 0.00  | 0.1045  | 0.1940  | 0.2537  | 0.8507  |
| casual    | 17379.0 | 31.158812  | 34.813147   | 0.00  | 4.0000  | 17.0000 | 48.0000 | 114.0000|
| registered| 17379.0 | 153.786869 | 151.357286  | 0.00  | 34.0000 | 115.0000| 220.0000| 886.0000|
| cnt       | 17379.0 | 189.463088 | 181.387599  | 1.00  | 40.0000 | 142.0000| 281.0000| 977.0000|



</div>

</td></tr> </table>


<div style="text-align: center;">
    <img src="docs/images/target_dist.png" style="max-width: 100%; height: auto;">
</div>


#### Problem, Goal and Solving approach
- This is a __regression__ problem  that uses the a bank dataset [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)  from kaggle to __predict the rented bike counts__ based on 17 features which are 8 categorical and 9 numerical.
- The __goal__ is to build a model that accurately __predict the rented bike counts__ based on the features.
- __Solving approach__ is that using the supervised machine learning models (linear, non-linear, ensemly).

#### Study
The project aimed predict the house prices using the features. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset.
- __(C) Pre-processing__: It includes data type casting, missing value handling, outlier handling.
- __(D) Exploratory Data Analysis__: Univariate, Bivariate, Multivariate anaylsises. Correlation and other relations. 
- __(E) Modelling__: Model tuning via GridSearch on Linear, Non-linear, Ensemble Models.  
- __(F) Saving the project__: Saving the project and demo studies.
- __(G) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __catboosting regression__ because of the results and less complexity.
<div style="flex: 50%; padding-left: 80px;">

|            | MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE          | RMSE         | MAE          | R2          | ExplainedVariance |
|----------- |-----------|------------------|-----------------------------|-------------|-------------|-------------|-------------|-------------------|
| cb     |   394.0 |             14.0 |                   38.254069 | 1668.109 | 40.84249 | 24.49770 |  0.950362 |          0.950362 |


</div>


- Model tuning results are below.

<table>
<tr><th>Linear Model</th></tr>
<tc><td>

| models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE      | RMSE    | MAE      | R2        | ExplainedVariance |
|-----------|----------|------------------|-----------------------------|----------|---------|----------|-----------|-------------------|
| lin_reg   | 0.387854 | 106.111623       | 20316.508631               | 649.0    | 82.0    | 343.15311 | 0.387854 | 142.535991        |
| l1_reg    | 649.0    | 82.0             | 343.060942                 | 20317.065880 | 142.537945 | 106.108458 | 0.387837 | 0.387837          |
| l2_reg    | 649.0    | 82.0             | 343.014057                 | 20315.509781 | 142.532487 | 106.105293 | 0.387884 | 0.387884          |
| enet_reg  | 648.0    | 82.0             | 342.883958                 | 20312.021289 | 142.520249 | 106.102992 | 0.387989 | 0.387991          |




</td><td> </table>


<table>
<tr><th>Non-Linear Model</th></tr>
<tc><td>

| models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE       | MAE        | R2         | ExplainedVariance |
|-----------|----------|-------------------|------------------------------|-------------|------------|------------|------------|-------------------|
| knn_reg   | 375.0    | 18.0              | 39.408859                    | 2707.928654 | 52.037762 | 31.837169 | 0.918409   | 0.918671          |
| svr_reg   | 671.0    | 34.0              | 78.921862                    | 13725.668585| 117.156599| 68.708285 | 0.586439   | 0.608872          |
| dt_params | 543.0    | 17.0              | 36.896549                    | 2919.705121 | 54.034296 | 31.352992 | 0.912028   | 0.912043          |



</td><td> </table>


<table>
<tr><th>Ensemble Model</th></tr>
<tc><td>

| models | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError |      MSE |     RMSE |      MAE |        R2 | ExplainedVariance |
|--------|---------|------------------|------------------------------|----------|----------|----------|-----------|-------------------|
| rf     |   371.0 |             14.0 |                   29.631831 | 2042.206 | 45.19077 | 26.08199 |  0.939230 |          0.939243 |
| gbr    |   383.0 |             23.5 |                   69.352007 | 3321.348 | 57.63114 | 38.06588 |  0.901167 |          0.901167 |
| xgbr   |   388.0 |             15.0 |                   34.368587 | 1719.381 | 41.46542 | 25.00144 |  0.948837 |          0.948837 |
| lgbm   |   411.0 |             14.0 |                   35.085998 | 1720.200 | 41.47530 | 24.81530 |  0.948812 |          0.948813 |
| cb     |   394.0 |             14.0 |                   38.254069 | 1668.109 | 40.84249 | 24.49770 |  0.950362 |          0.950362 |


</td><td> </table>


## Details

### Abstract
- [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) is used to predict the purchase amount. The dataset has 550068 records, 12 features which are 8 categorical and 4 numerical typed. The problem is supervised learning task as regression. The goal is predicting  a purchase value  correctly through using supervised machine learning algorithms such as non-linear, ensemble and similar model.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through cross validation and Grid Search model tuning approachs. Hyperparameter tuning implemented Greedy Greed Search approach which tunes a hyper param at once a time while iterating the sorted order according the importance of the hyperparams. Models are evaluated with cross validation methods using 5 split. Regression results collected and compared between the models. Selected the basic and more succesful model. Tuned __lgbm regression__ model has __2959.133388__ RMSE , __2228.421937__ MAE, __0.655337__ R2, __0.655338__ Explained Variance, the other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── component_configs.json
│   ├── requirements.txt
│   ├── finalized.sav
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/component_configs.json :
    - It includes the web components to generate web page.
  - demo_app/finalized.sav:
    - The trained (Model Tuned) model as pickle (python object saving) format.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    


### Explanation of the Study
#### __(A) Dependencies__:
  -  There is a third-parth installation which is kaggle dataset api, just follow the study codes it will be handled. The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
#### __(B) Dataset__: 
  - Downloading the [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday) via kaggle dataset api from kaggle platform. The dataset has 550068 records. There are 12 features which are 8 categorical and 4 numerical typed. For more info such as histograms and etc... you can look the '(D) Exploratory Data Analysis' chapter.
#### __(C) Pre-processing__: 
  - The processes are below:
    - Preparing the dtypes such as casting the object type to categorical type.
    - Missing value processes: Finding the missing values and handled the missing value via dropping or imputation.
    - Outlier analysis processes: uses  both visual and IQR calculation apporachs. According to IQR approach, detected statistically significant outliers are handled using boundary value casting assignment method. (There was no outlier value as statistically significant)

      <div style="text-align: center;">
          <img src="docs/images/outliers.png" style="width: 600px; height: 150px;">
      </div>
 
#### __(D) Exploratory Data Analysis__:
  - Dataset Stats
<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>


| #   | Column      | Non-Null Count | Dtype    |
| --- | ----------- | -------------- | -------- |
| 0   | instant     | 17379 non-null | int64    |
| 1   | dteday      | 17379 non-null | object   |
| 2   | season      | 17379 non-null | int64    |
| 3   | yr          | 17379 non-null | int64    |
| 4   | mnth        | 17379 non-null | int64    |
| 5   | hr          | 17379 non-null | int64    |
| 6   | holiday     | 17379 non-null | int64    |
| 7   | weekday     | 17379 non-null | int64    |
| 8   | workingday  | 17379 non-null | int64    |
| 9   | weathersit  | 17379 non-null | int64    |
| 10  | temp        | 17379 non-null | float64  |
| 11  | atemp       | 17379 non-null | float64  |
| 12  | hum         | 17379 non-null | float64  |
| 13  | windspeed   | 17379 non-null | float64  |
| 14  | casual      | 17379 non-null | int64    |
| 15  | registered  | 17379 non-null | int64    |
| 16  | cnt         | 17379 non-null | int64    |

</td><td>

<div style="flex: 50%; padding-left: 50px;">

|      |    count |        mean |         std |   min |      25% |      50% |      75% |      max |
|:-----|---------:|------------:|------------:|------:|---------:|---------:|---------:|---------:|
|instant     | 17379 |  8690.00000 | 5017.029500 | 1.00 | 4345.5000 | 8690.0000 | 13034.5000 | 17379.0000 |
|season      | 17379 |     2.50164 |    1.106918 | 1.00 |    2.0000 |    3.0000 |    3.0000 |    4.0000 |
|yr          | 17379 |     0.50256 |    0.500008 | 0.00 |    0.0000 |    1.0000 |    1.0000 |    1.0000 |
|mnth        | 17379 |     6.53778 |    3.438776 | 1.00 |    4.0000 |    7.0000 |   10.0000 |   12.0000 |
|hr          | 17379 |    11.54675 |    6.914405 | 0.00 |    6.0000 |   12.0000 |   18.0000 |   23.0000 |
|holiday     | 17379 |     0.02877 |    0.167165 | 0.00 |    0.0000 |    0.0000 |     0.0000 |     1.0000 |
|weekday     | 17379 |     3.00368 |    2.005771 | 0.00 |    1.0000 |    3.0000 |     5.0000 |     6.0000 |
|workingday  | 17379 |     0.68272 |    0.465431 | 0.00 |    0.0000 |    1.0000 |     1.0000 |     1.0000 |
|weathersit  | 17379 |     1.42528 |    0.639357 | 1.00 |    1.0000 |    1.0000 |     2.0000 |     4.0000 |
|temp        | 17379 |     0.49699 |    0.192556 | 0.02 |    0.3400 |    0.5000 |     0.6600 |     1.0000 |
|atemp       | 17379 |     0.47578 |    0.171850 | 0.00 |    0.3333 |    0.4848 |     0.6212 |     1.0000 |
|hum         | 17379 |     0.62723 |    0.192930 | 0.00 |    0.4800 |    0.6300 |     0.7800 |     1.0000 |
| windspeed | 17379.0 | 0.190098   | 0.122340    | 0.00  | 0.1045  | 0.1940  | 0.2537  | 0.8507  |
| casual    | 17379.0 | 31.158812  | 34.813147   | 0.00  | 4.0000  | 17.0000 | 48.0000 | 114.0000|
| registered| 17379.0 | 153.786869 | 151.357286  | 0.00  | 34.0000 | 115.0000| 220.0000| 886.0000|
| cnt       | 17379.0 | 189.463088 | 181.387599  | 1.00  | 40.0000 | 142.0000| 281.0000| 977.0000|

</div>

</td></tr> </table>
  - Variable Analysis
    - Univariate analysis, 
      <div style="text-align: center;">
          <img src="docs/images/feat_dists.png" style="width: 400px; height: 200px;">
          <img src="docs/images/feat_violin.png" style="width: 400px; height: 200px;">
      </div>
    - Bivariate analysis
      <div style="text-align: center;">
          <img src="docs/images/bi_var_1.png" style="width: 400px; height: 300px;">
          <img src="docs/images/bi_var_2.png" style="width: 400px; height: 300px;">
      </div>
    - Multivariate analysis.
      <div style="text-align: center;">
          <img src="docs/images/multi_1.png" style="width: 400px; height: 300px;"> 
          <img src="docs/images/multi_2.png" style="width: 400px; height: 300px;"> 
          <img src="docs/images/multi_3.png" style="width: 400px; height: 300px;"> 
      </div>
  - Other relations.
    <div style="display:flex; justify-content: center; align-items:center;">
      <div style="text-align: center;">
      <figure>
      <p>Correlation</p>
      <img src="docs/images/corr_heat_map.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
       <div style="text-align: center;">
      <figure>
      <p>Correlation between target</p>
      <img src="docs/images/corrs_feat.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Variance</p>
      <img src="docs/images/feat_var.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Covariance</p>
      <img src="docs/images/feat_covar.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
    </div>

#### __(E) Modelling__: 
  - Data Split
    - Splitting the dataset via  sklearn.model_selection.train_test_split (test_size = 0.2).
  - Util Functions
    - Greedy Step Tune
      - It is a custom tuning approach created by me. It tunes just a hyperparameter per step using through GridSerchCV. It assumes the params ordered by importance so it reduces the computation and time consumption.  
    - Model Tuner
      - It is an abstraction of the whole training process. It aims to reduce the code complexity. It includes the corss validation and GridSerachCV approachs to implement training process.
    - Learning Curve Plotter
      - Plots the learning curve of the already trained models to provide insight.
  - Linear Model Tuning Results _without balanciy process_
    - linear, l1, l2, enet regressions
    - Cross Validation Scores
      | models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE      | RMSE    | MAE      | R2        | ExplainedVariance |
      |-----------|----------|------------------|-----------------------------|----------|---------|----------|-----------|-------------------|
      | lin_reg   | 0.387854 | 106.111623       | 20316.508631               | 649.0    | 82.0    | 343.15311 | 0.387854 | 142.535991        |
      | l1_reg    | 649.0    | 82.0             | 343.060942                 | 20317.065880 | 142.537945 | 106.108458 | 0.387837 | 0.387837          |
      | l2_reg    | 649.0    | 82.0             | 343.014057                 | 20315.509781 | 142.532487 | 106.105293 | 0.387884 | 0.387884          |
      | enet_reg  | 648.0    | 82.0             | 342.883958                 | 20312.021289 | 142.520249 | 106.102992 | 0.387989 | 0.387991          |

    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/linear_regression_feat_imp.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_reg_feat_imp.png" style="width: 450px; height: 200px;">
      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/linear_regression_l_cur.png" style="width: 150px; height: 200px;">
          <img src="docs/images/ling_reg_l_cur.png" style="width: 450px; height: 200px;">
      </div>
  - Non-Linear Models
    - Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Tree
    - Cross Validation Scores _without balanciy process_
      | models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE       | MAE        | R2         | ExplainedVariance |
      |-----------|----------|-------------------|------------------------------|-------------|------------|------------|------------|-------------------|
      | knn_reg   | 375.0    | 18.0              | 39.408859                    | 2707.928654 | 52.037762 | 31.837169 | 0.918409   | 0.918671          |
      | svr_reg   | 671.0    | 34.0              | 78.921862                    | 13725.668585| 117.156599| 68.708285 | 0.586439   | 0.608872          |
      | dt_params | 543.0    | 17.0              | 36.896549                    | 2919.705121 | 54.034296 | 31.352992 | 0.912028   | 0.912043          |
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/non_lin_l_cur.png" style="width: 400px; height: 300px;">
      </div>


  - Ensemble Models
    - Random Forest, Gradient Boosting Machines, XGBoost, LightGBoost, CatBoost
    - Cross Validation Scores _without balanciy process_
      | models | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError |      MSE |     RMSE |      MAE |        R2 | ExplainedVariance |
      |--------|---------|------------------|------------------------------|----------|----------|----------|-----------|-------------------|
      | rf     |   371.0 |             14.0 |                   29.631831 | 2042.206 | 45.19077 | 26.08199 |  0.939230 |          0.939243 |
      | gbr    |   383.0 |             23.5 |                   69.352007 | 3321.348 | 57.63114 | 38.06588 |  0.901167 |          0.901167 |
      | xgbr   |   388.0 |             15.0 |                   34.368587 | 1719.381 | 41.46542 | 25.00144 |  0.948837 |          0.948837 |
      | lgbm   |   411.0 |             14.0 |                   35.085998 | 1720.200 | 41.47530 | 24.81530 |  0.948812 |          0.948813 |
      | cb     |   394.0 |             14.0 |                   38.254069 | 1668.109 | 40.84249 | 24.49770 |  0.950362 |          0.950362 |


    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_feat_imp.png" style="width: 800px; height: 200px;">

      </div>
      - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_l_cur.png" style="width: 800px; height: 200px;">

      </div>


#### __(F) Saving the project__: 
  - Saving the project and demo studies.
    - trained model __cb.sav__ as pickle format.
#### __(G) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is predicting the sales price based on four features.
    - Usage: Set the feature values through sliding the radio buttons and dropdown menu then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-bikesharingcountregression.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

