# Tutorial

## Prepare master data set

#### Step 1

* Put all the necessary independent variables and target variables into a single table. The modeling population for each target variable may not be the same. In that case, consolidate each data set to a single table using a full join.
* The toolbox can handle both categorical variables and numerical variables. A categorical variable will be converted into dummy variables for each label of the categorical variable.
* Missing value treamtment:
    * For categorical variables, you can specify a label to fill the missing values for all categorical variables.
    * For numerical variables, the mean value of each variable in train data set (after under-sampling, to be fixed in the future) will be used to impute the misssing values.
    * If you want to use other methods to impute missing values, please do so beforehand.
* All variable names must start with a prefix either `CAT_` or `NUM_` depending on the variable type. If not, you may see unexpected results. 

#### Step 2

##### Hive table for `prep_spark` command (large data set)

* Import your data to an external or internal Hive table with the appropriate schema. If you're unsure how to get your data into Hive, please reference this [link](https://docs.hortonworks.com/HDPDocuments/HDP2/HDP-2.3.0/bk_dataintegration/content/moving_data_from_hdfs_to_hive_external_table_method.html).

* Create a text file `colnames_TARGET.txt` for each target variable `TARGET`. The text file should include the name of independent variables to be considered for each target, one variable name per line.

##### CSV file for `prep` command (small data set)

* Export the master data set to a csv file without header row. If you happen to have a huge csv file with the header row, you can use the following Linux command to remove it.

```sh
$ tail -n+2 csv_with_header.csv > csv_without_header.csv
```

* If you use single node version of data preprocessing command `prep`, you can compress the csv file to reduce I/O time.

```sh
$ gzip csv_without_header.csv
```

* Create a text file `colnames.txt` that contains the name of all variables in the csv file. Place one variable name per line in order and make sure you follow the variable naming convention.
* Create a text file `colnames_TARGET.txt` for each target variable `TARGET`. The text file should include the name of independent variables to be considered for each target, one variable name per line.

#### Step 3

* For pandas version of data preprocessing command `prep`, place the gzipped csv file, header file and target-specific column name files into a single folder.
* For Spark version of data preprocessing command `prep_spark`, place target-specific column name files into a single folder and note your Hive database and table name.

## Prepare scoring data set

* Similar to master data set, prepare the scoring data set as a csv file without header row. 
* Make sure the scoring data set contains all the independent variables that are used in the best model for each target and a single id column that identifies a unique row. The target variables won't be needed.
* Prepare a text file that contains the name of all variables in the scoring data set.


## Prepare configuration file in `json` format

The modeling toolbox uses a single `json` file for configuration. A part of the configuration file can be commented with `//` or `#`. See below the list of options that you can specify.

#### Version

Specify the version of `mtbox` to use. It should match the version of the command line tool. 

```json
"version": "0.2.3"
```

To check the version of the command line tool:

```sh
(../py2-mtbox) $ mtbox --version
```

#### Input

Specify the folder that contains the input data, including target-specific column names (and zipped csv file and header file for pandas version of data preprocessing command `prep`). Either relative or absolute path can be used.

```json
"path_to_input": "input"
```

#####  Hive table

This option is used for Spark version of data preprocessing command `prep_spark`. Specify the hive database name and table name under the `hive_input_data` key.


```json
"hive_input_data": {
        "db_name": "hcaadvaph_wk_mtbox",
        "table_name": "input"
        },
```

#####  CSV file

This option is used for pandas version of data preprocessing command `prep`. Specify the master data set file name. You do not need to include the full path to the file. The toolbox will search the master data set file in the folder specified in `path_to_input`.

```json
"data_filename": "hcc_data.csv.gz"
```

Specify the separator for (gzipped) csv file.

```json
"separator": ","
```

(DEPRECATED) Specify the master data set file name on HDFS. This option is used for Spark version of data preprocessing command `prep_spark`. Use the absolute path to the file.

```json
"data_filename_on_hdfs": "/hdfsdata/vs1/hca/adva/phi/all_lob/work/mtbox/example/input"
```

#### Targets

Specify target variables to build models for by listing the target variable names.

```json
"targets": [
    "NUM_HCC2015_142",
    "NUM_HCC2015_88"
]
```

#### Output

Specify the folder to write outputs - preprocessed data, feature importance, model objects, predictions, model reports and test results.

```json
"path_to_output": "output"
```

#### Number of cores

Specify the number of cores to use for parallel processes.

```json
"n_jobs": 16
```

#### Verbosity

Specify the verbosity of the modeling toolbox log messages. 

```json
"verbose": 1
```

The option `"verbose"` can take the following values.
* `0`: silent
* `1`: standard progress messages
* `2`: detailed messages

#### Random seed

Specify the seed for random number generators.

```json
"random_state": 0
```

#### Missing value treatment

Specify the label to use impute categorical missing values.

```json
"fill_values_categorical_missing": "UNK"
```

#### Sampling

Specify the size of hold-out test data set as a proportion to the entire data set (between 0 and 1).

```json
"test_size": 0.3
```

Specify the event rate in the train data set to be undersampled (between 0 and 1). If you do not want to undersample, then set it as `null`.

```json
"train_event_rate": 0.25
```

You can also specify different event rate for a certain target by providing the detail as a json string. When doing so, you must have "default" key and the default train event rate to use for all other targets.

```json
 "train_event_rate": {
        "default": 0.25,
        "NUM_HCC2015_142": 0.3
    }
```


#### Feature importance

Specify methods to explore independent variables and compute feature importance for feature selection. You can list multiple methods to explore features using the following template. The command `explore` will use the feature exploration methods specified in this option to compute feature importance. This is optional. If not specified, `explore` command will be skipped and `build` command will use all features provided in `colnames_TARGET.txt`. 

```json
"feature_importance": {
    "METHOD_NAME_1" : {
        "method": "METHOD_1",
        "params": {
            ...
        }
    },
    "METHOD_NAME_2" : {
        "method": "METHOD_2",
        "params": {
            ...
        }
    }
}
```

Currently, random forest model is available for feature exploration, i.e. the option `"method"` can take value `"random_forest"` only. 

```json
"feature_importance": {
    "random_forest" : {
        "method": "random_forest",
        "params": {
            "n_estimators": 300,
            "min_samples_split": 50,
            "min_samples_leaf": 30
        }
    }
}
```

See scikit-learn documentation on [Random Forest classifier] for the parameters to use for `"params"`. The following parameters are specified by the toolbox and you cannot change them.

* n_jobs
* verbose
* random_state

#### Feature selection

Specify the feature exploration method for the feature selection to be based on in `build` command. This is optional. If not specified, `build` command will use all features provided in `colnames_TARGET.txt`.

```json
"feature_selection": {
    "feature_importance": "random_forest",
    "which": "top",
    "threshold": 50
}
```

Specify the name of the feature exploration method you want to use among the method names in `feature_importance` option.

The option `"which"` can take the following values.

* `"top"`: select top `"threshold"` many variables. When there are ties, the number of selected variables will not exceed `"threshold"`.
* `"at_least"`: select variables with feature importance being greater than or equal to `"threshold"` value (between 0 and 1).
* `"coverage"`: select top variables of which the cumulative feature importance is less than or equal to `"threshold"` value (between 0 and 1). 

#### Validation

Specify the number of folds to use in cross-validated grid search and cross-validation score.

```json
"n_folds": 3
```

Specify the evaluation metric to use in cross-validated grid search and model performance evaluation in train, cross-validation and test data sets.

```json
"eval_metric": "roc_auc"
```

The option `"eval_metric"` can take the following values.

* `"roc_auc"`: average under Receiver-Operating-Characteristic (ROC) curve.
* `"average_precision"`: area under Precision-Recall (PR) curve.
* `"log_loss"`: log loss, i.e. logistic loss or cross-entropy loss.

#### Models

Specify models to build. You can list multiple models to build using the following template.

```json
"models": {
    "MODL_NAME_1": {
        "method": "METHOD_1",
        "params": {
            ...
        }
    },
    "MODL_NAME_2": {
        "method": "METHOD_2",
        "params": {
            ...
        },
        "params_to_search": {
            ...
        },
        "feature_selection": {
            ...
        }
    }
}
```

The option `"method"` can take the following values.

* `"logit"`: scikit-learn linear_model.LogisticRegression model (optional l1, l2 regularization)
* `"logit_sgd"`: regularized logistic regression from scikit-learn (via stochastic gradient descent)
* `"random_forest"`: random forest model from scikit-learn
* `"xgboost"`: gradient boosted trees model from xgboost model
 
Specify model parameters to use for all targets in `"params"`.

```json
"params": {
    "n_estimators": 100,
    "min_child_weight": 30,
    "subsample": 1.0,
    "colsample_bytree": 1.0
}
```

* For logistic regression model, see scikit-learn documentation on [Logistic Regression classifier]. The following parameters are specified by the toolbox and you cannot change them.
    * random_state
    * verbose
    * n_jobs
    * warm_start
* For logistic regression model based on Stochastic Gradient Descent (SGD) algorithm, see scikit-learn documentation on [SGD classifier]. The following parameters are specified by the toolbox and you cannot change them.
    * loss
    * random_state
    * verbose
    * n_jobs
    * warm_start
* For random forest model, see scikit-learn documentation on [Random Forest classifier]. The following parameters are specified by the toolbox and you cannot change them.
    * oob_score
    * warm_start
    * class_weight
    * verbose
    * n_jobs
    * random_state
* For gradient boosted trees model, see docstrings in the source code for [XGBModel class]. The following parameters are specified by the toolbox and you cannot change them.
    * objective
    * silent
    * nthread
    * seed

* If building logistic regression models, mtbox will build a comparable model with [StatsModels] for evaluation (L2 not available).
* Note that the feature importance report for logistic regression models is not meaningful. The estimated coefficient will be reported in the relative feature importance column, but the cumulative feature importance column won't be useful.
    
Specify model parameters and a list of values for each parameter to explore in grid search in `"params_to_search"`. This is optional. If not specified, the model will be fitted with fixed parameters specified in `"params"` (which is also optional, if you wish to use the default parameters).

```json
"params_to_search": {
    "learning_rate": [0.1, 0.3],
    "max_depth": [4, 6]
}
```

Specify final feature selection strategy in `"feature_selection"`. After the initial model is built, the final variables will be selected and the final model will be re-fitted with the same (or the best, if grid search is done) parameters. This is optinoal. Use the same options as the main feature selection option described in earlier section. Currently, this option is disabled for logistic regression models, since there is no relative feature importance measure in logistic regression.

```json
"feature_selection": {
    "which": "coverage",
    "threshold": 0.95
}
```

For xgboost models, you can optionally configure early stopping option. If specified, xgboost models will stop building additional trees when the model performance on the validation set does not improve. It will use `"eval_metric"` for performance measure.

```json
"early_stopping": {
    "during_grid_search": true,
    "early_stopping_rounds": 5,
    "eval_set_size": 0.3
}
```

* `"during_grid_search"`: apply early stopping during grid search (true or false).
* `"early_stopping_rounds"`: validation error needs to decrease at least every `"early_stopping_rounds"` round(s) to continue training.
* `"eval_set_size"`: the size of validation data set as a proportion to the train data set (between 0 and 1).

See a complete example for `"models"` option below.

```json
"models": {
    "logit": {
        "method": "logit",
        "params": {
            "penalty": "l2", // or "l1"
        },
        "params_to_search": {
            "C": [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001] // Higer values ~ no regularization
        }
    },
    "logit_sgd": {
        "method": "logit_sgd",
        "params": {
            "penalty": "l1",
            "n_iter": 5
        },
        "params_to_search": {
            "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001]
        }
    },
    "random_forest": {
        "method": "random_forest",
        "params": {
            "n_estimators": 100,
            "min_samples_split": 50,
            "min_samples_leaf": 30,
            "bootstrap": false
        },
        "params_to_search": {
            "max_depth": [8, 12]
        },
        "feature_selection": {
            "which": "coverage",
            "threshold": 0.95
        }
    },
    "xgboost": {
        "method": "xgboost",
        "params": {
            "n_estimators": 100,
            "min_child_weight": 30,
            "subsample": 1.0,
            "colsample_bytree": 1.0
        },
        "params_to_search": {
            "learning_rate": [0.1, 0.3],
            "max_depth": [4, 6]
        },
        "feature_selection": {
            "which": "coverage",
            "threshold": 0.95
        }
    },
    "xgboost_early_stopping": {
        "method": "xgboost",
        "params": {
            "n_estimators": 100,
            "min_child_weight": 30,
            "subsample": 1.0,
            "colsample_bytree": 1.0
        },
        "params_to_search": {
            "learning_rate": [0.1, 0.3],
            "max_depth": [4, 6]
        },
        "feature_selection": {
            "which": "coverage",
            "threshold": 0.95
        },
        "early_stopping": {
            "during_grid_search": true,
            "early_stopping_rounds": 5,
            "eval_set_size": 0.3
        }
    }
},
```

#### Manifold learning

Specify maxinum number of samples from preprocessed train data to use in manifold learning. The t-SNE algorithm is computationally expensive and you may want to limit the number of samples. The train data set with more rows than the set value will be randomly sampled (stratified, without replacement).

```json
"max_num_samples_for_manifold_learning": 10000,
```

Specify manifold learning models (t-SNE) to build. You can list multiple models to build using the following template.

```json
"manifold_learning": {
    "MODL_NAME_1": {
        "method": "METHOD_1",
        "params": {
            ...
        }
    },
    "MODL_NAME_2": {
        "method": "METHOD_2",
        "params": {
            ...
        }
    }
}
```

The option `"method"` can take the following values.

* `"tsne"`: Barnes-Hut t-SNE implementation by Dmitry Ulyanov
* `"sk_tsne"`: t-SNE implementation from scikit-learn

Specify model parameters to use for all targets in `"params"`.

```json
"params": {
    "perplexity": 30.0,
    "n_iter": 1000,
    "angle": 0.5
}
```

* For Barnes-Hut t-SNE implementation by Dmitry Ulyanov, see his Github repository [Multicore t-SNE]. The following parameters are specified by the toolbox and you cannot change them.
    * n_jobs (Note that the parameter n_jobs is set to 1 by the toolbox and the parallel implementation of the t-SNE algorithm in this package is disabled. The jobs will be parallelized at target / manifold learning level instead.)
* For Barnes-Hut t-SNE implementation by Dmitry Ulyanov, you can only change the following parameters.
    * perplexity
    * n_iter
    * angle
* For t-sNE implementation from scikit-learn, see scikit-learn documentation on [t-SNE]. The following parameters are specified by the toolbox and you cannot change them.
    * verbose
    * random_state
    * n_components

See a complete example for `"manifold_learning"` option below.

```json
"manifold_learning": {
    "tsne": {
        "method": "tsne",
        "params": {
            "perplexity": 30.0,
            "n_iter": 1000,
            "angle": 0.5
        }
    },
    "sk_tsne": {
        "method": "sk_tsne",
        "params": {
            "perplexity": 30.0,
            "n_iter": 1000,
            "learning_rate": 1000,
            "angle": 0.5,
            "init": "pca"
        }
    }
},
```

#### Scoring

Specify column name for identification column (unique primary key).

```json
"colname_for_id": "CAT_MCID"
```

##### Scoring data in Hive table

Specify the Hive database and table name for the scoring input and output data.

In order to activate the Hive portion of the `score` function, you must provide the `hive_scoring_data` key in the `conf.json`.

```json
"hive_scoring_data": {
        "input_db_name": "mtbox_example",
        "input_table_name": "201610_scoring_data",
        "output_db_name":"mtbox_example",
        "output_table_name":"201610_scores"
}
```

Do additional analysis on scoring output. This adds rank and percent rank to all scores.

0 = OFF : 1 = ON

```json
"score_analysis_flag": 1
```

##### Scoring data in CSV on HDFS (deprecated)

Specify the scoring data set file name on HDFS.

```json
"scoring_data_filename_on_hdfs": "/hdfsdata/vs1/hca/adva/phi/all_lob/work/mtbox/example/scoring_data"
```

Specify the column names in scoring data set.

```json
"scoring_data_colnames": "score/colnames_201507.txt"
```

Specify path to write final scores on HDFS.

```json
"path_to_scores_on_hdfs": "/hdfsdata/vs1/hca/adva/phi/all_lob/work/mtbox/example/score"
```

Specify the number of RDD partitions for scoring data. This is optional.

```json
"num_partitions_scoring_data": 2000
```

Specify the number of RDD partitions for final scores.

```json
"num_partitions_final_scores": 5
```

#### PySpark

Specify PySpark options for `prep_spark` command.

```json
"pyspark": {
    "enviornment_variables": {
        "PYSPARK_DRIVER_PYTHON": "/app/hca/adva/mtbox/py2-mtbox/bin/python",
        "PYSPARK_PYTHON": "/opt/cloudera/parcels/Anaconda/bin/python"
    },
    "pyspark_bin": "pyspark",
    "options": {
        "--master": "yarn",
        "--num-executors": 150,
        "--executor-memory": "2G",
        "--driver-memory": "4G",
        "--conf": {
            "spark.driver.maxResultSize": "3G"
        }
    }
}
```

#### Dashboard

Specify dashboard options for `dashboard` command. Use the option `"port"` to specify the port number for the modeling toolbox dashboard. This is optional. If not specified, a random open port will be assigned.

```json
"dashboard": {
    "port": 7619
}
```

## Use `mtbox` command line tools

If all of the above configuration is compiled into a sinle `json` file, i.e. as `conf.json` file, then we can use the command line tools to build models.

#### `prep` or `prep_spark`

The command `prep` will preprocess input data to prepare train and test data set.

```sh
(../py2-mtbox) $ mtbox conf.json prep
```

The command `prep_spark` will preprocess input data to prepare train and test data set with Spark.

```sh
(../py2-mtbox) $ mtbox conf.json prep_spark
```

* Read input data.
* Impute missing values for categorical variables and numerical variables.
* Convert categorical variables to numerical labels and to dummy variables.
* Split input data into train and test data sets.
* Undersample train data to the specified event rate.
* Export train and test data sets in sparse matrix format for fast I/O (`"path_to_output"/preprocessed_data/`).
* Export variable transformation information (`"path_to_output"/scoring/`).

#### `explore`

The command `explore` will explore independent variables and compute feature importance.

```sh
(../py2-mtbox) $ mtbox conf.json explore
```

* Read train data.
* Explore features with specified method and genreate feature importance report (`"path_to_output"/feature_importance/`).

#### Manual adjustment to feature selection

You can modify the feature importance report generated by `explore` to force certain variables to be included or excluded in the subsequent `build` stage.

* To include a variable, change 'rank' of the variable to be 0 from the feature importance report.
* To exclude a variable, change 'rank' of the variable to be -1 from the feature importance report. 

#### `build`

The command `build` will build models with train data.

```sh
(../py2-mtbox) $ mtbox conf.json build
```

* Read train data.
* Select features based on the feature importance report and the specified feature selection strategy.
* Build models with fixed parameter or grid search.
* Finalize models if final feature selection strategy is specfied for each model.
* Export model objects (`"path_to_output"/models/`).
* Export predictions for train data (`"path_to_output"/predicted/`).
* Export model reports (`"path_to_output"/reports/`).
* Export a list of best model for each target and all variables to be used in `score` command (`"path_to_output"/scoring/`).

#### `test`

The command `test` will test models with test data.

```sh
(../py2-mtbox) $ mtbox conf.json test
```

* Read test data and model objects.
* Score test data using model objects.
* Export predictions for test data (`"path_to_output"/predicted/`).
* Export test results (`"path_to_output"/reports/`).

#### `manifold`

The command `manifold` will build manifold learning models (t-SNE) with train data.

```sh
(../py2-mtbox) $ mtbox conf.json manifold
```

* Read train data.
* Select features based on the feature importance report and the specified feature selection strategy (`"feature_selection"` option).
* Sub-sample from train data if the maxinum number of samples for manifold learning is specified.
* Build manifold learning models specified in `"manifold_learning"` option.
* Export embeddings (`"path_to_output"/embedding`).
* Export train data used in manifold learning model with selected features and samples (`"path_to_output"/embedding/data_for_manifold_learning`).

#### `dashboard`

The command `dashboard` will show various model performance charts and univariate analysis charts in a web dashboard. Go to the address shown at the end of the screen output to access the dashboard.

```sh
(../py2-mtbox) $ mtbox conf.json dashboard
```

* Display feature importance from `explore` command.
* Visualize embeddings from manifold learning models and allow you to select segments and profile them.
    * Each point in the plot represents a sample in (sub-sampled) train data.
    * The color of points scales between blue and red for actual value of target variable and the prediction score of the selected model. (0 means non-event and blue. 1 means event and red.)
    * The color of points scales between yellow, orange and red for log loss of prediction error. (Higher the log loss error, more red.)
    * You can lasso-select or box-select points of interest and profile the selected points. It will run t-test for each features selected by `"feature_selection"` option and sort the features by p-value of the t-test.
* Display feature importance table and plot univariate analysis chart for important features.
* Plot various model performance tables and charts.

#### `score`

The command `score` will score new data set with best model for each target.

```sh
(../py2-mtbox) $ mtbox conf.json score
```

* Read scoring data set from Hive table and model objects from `"path_to_output"/models`.
* Score the data set using model objects.
* Export score to a Hive table.

  [Logistic Regression classifier]: <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
  [SGD classifier]: <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>
  [StatsModels]: <http://statsmodels.sourceforge.net/stable/glm.html>
  [Random Forest classifier]: <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
  [XGBModel class]: <https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py>
  [Multicore t-SNE]: <https://github.com/DmitryUlyanov/Multicore-TSNE>
  [t-SNE]: <http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>
  
  
