
I want a script that compares many different clustering algorithms on the same dataset. I wan the results of each in a single combined dataset. 

I then want to choose an algo from this list for the combined dataset and then do analysis on it, clustering ect.

# check installed version
import pycaret
pycaret.__version__

# # 🚀 Quick start


# PyCaret's Clustering Module is an unsupervised machine learning module that performs the task of grouping a set of objects in such a way that objects in the same group (also known as a cluster) are more similar to each other than to those in other groups. 
# 
# It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 10 ready-to-use algorithms and several plots to analyze the performance of trained models. 
# 
# A typical workflow in PyCaret's unsupervised module consist of following 6 steps in this order:
# 
# ### **Setup** ➡️ **Create Model** ➡️ **Assign Labels** ➡️ **Analyze Model** ➡️ **Prediction** ➡️ **Save Model**


# loading sample dataset from pycaret dataset module
from pycaret.datasets import get_data
data = get_data('jewellery')


# ## Setup
# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function in PyCaret. It only has one required parameter i.e. `data`. All the other parameters are optional.


# import pycaret clustering and init setup
from pycaret.clustering import *
s = setup(data, session_id = 123)


# Once the setup has been successfully executed it shows the information grid containing experiment level information. 
# 
# - **Session id:**  A pseudo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions.<br/>
# <br/>
# - **Original data shape:**  Shape of the original data prior to any transformations. <br/>
# <br/>
# - **Transformed data shape:**  Shape of data after transformations <br/>
# <br/>
# - **Numeric features :**  The number of features considered as numerical. <br/>
# <br/>
# - **Categorical features :**  The number of features considered as categorical. <br/>


# PyCaret has two set of API's that you can work with. (1) Functional (as seen above) and (2) Object Oriented API.
# 
# With Object Oriented API instead of executing functions directly you will import a class and execute methods of class.


# import ClusteringExperiment and init the class
from pycaret.clustering import ClusteringExperiment
exp = ClusteringExperiment()


# check the type of exp
type(exp)


# init setup on exp
exp.setup(data, session_id = 123)


# You can use any of the two method i.e. Functional or OOP and even switch back and forth between two set of API's. The choice of method will not impact the results and has been tested for consistency.
# ___


# ## Create Model
# 
# This function trains and evaluates the performance of a given model. Metrics evaluated can be accessed using the `get_metrics` function. Custom metrics can be added or removed using the `add_metric` and `remove_metric` function. All the available models can be accessed using the `models` function.


# train kmeans model
kmeans = create_model('kmeans')


# to check all the available models
models()


# train meanshift model
meanshift = create_model('meanshift')


# ## Assign Model
# This function assigns cluster labels to the training data, given a trained model.


kmeans_cluster = assign_model(kmeans)
kmeans_cluster


# ## Analyze Model


# You can use the `plot_model` function to analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.


# plot pca cluster plot 
plot_model(kmeans, plot = 'cluster')


# plot elbow
plot_model(kmeans, plot = 'elbow')


# plot silhouette
plot_model(kmeans, plot = 'silhouette')


# check docstring to see available plots 
# help(plot_model)


# An alternate to `plot_model` function is `evaluate_model`. It can only be used in Notebook since it uses ipywidget.


evaluate_model(kmeans)


# ## Prediction
# The `predict_model` function returns `Cluster` label as a new column in the input dataframe. This step may or may not be needed depending on the use-case. Some times clustering models are trained for analysis purpose only and the interest of user is only in assigned labels on the training dataset, that can be done using `assign_model` function. `predict_model` is only useful when you want to obtain cluster labels on unseen data (i.e. data that was not used during training the model).


# predict on test set
kmeans_pred = predict_model(kmeans, data=data)
kmeans_pred


# ## Save Model


# Finally, you can save the entire pipeline on disk for later use, using pycaret's `save_model` function.


# save pipeline
save_model(kmeans, 'kmeans_pipeline')


# load pipeline
kmeans_pipeline = load_model('kmeans_pipeline')
kmeans_pipeline


# # 👇 Detailed function-by-function overview


# ## ✅ Setup
# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function in PyCaret. It only has one required parameter i.e. `data`. All the other parameters are optional.


# init setup
s = setup(data, session_id = 123)


# To access all the variables created by the setup function such as transformed dataset, random_state, etc. you can use `get_config` method.


# check all available config
get_config()


# lets access X_train_transformed
get_config('X_train_transformed')


# another example: let's access seed
print("The current seed is: {}".format(get_config('seed')))

# now lets change it using set_config
set_config('seed', 786)
print("The new seed is: {}".format(get_config('seed')))


# All the preprocessing configurations and experiment settings/parameters are passed into the `setup` function. To see all available parameters, check the docstring:


# help(setup)


# init setup with normalize = True

s = setup(data, session_id = 123,
          normalize = True, normalize_method = 'minmax')


# lets check the X_train_transformed to see effect of params passed
get_config('X_train_transformed')['Age'].hist()


# Notice that all the values are between 0 and 1 - that is because we passed `normalize=True` in the `setup` function. If you don't remember how it compares to actual data, no problem - we can also access non-transformed values using `get_config` and then compare. See below and notice the range of values on x-axis and compare it with histogram above.


get_config('X_train')['Age'].hist()


# ## ✅ Experiment Logging
# PyCaret integrates with many different type of experiment loggers (default = 'mlflow'). To turn on experiment tracking in PyCaret you can set `log_experiment` and `experiment_name` parameter. It will automatically track all the metrics, hyperparameters, and artifacts based on the defined logger.


# from pycaret.clustering import *
# s = setup(data, log_experiment='mlflow', experiment_name='jewellery_project')


# train kmeans
# kmeans = create_model('kmeans')


# start mlflow server on localhost:5000
# !mlflow ui


# By default PyCaret uses `MLFlow` logger that can be changed using `log_experiment` parameter. Following loggers are available:
#     
#     - mlflow
#     - wandb
#     - comet_ml
#     - dagshub
#     
# Other logging related parameters that you may find useful are:
# 
# - experiment_custom_tags
# - log_plots
# - log_data
# - log_profile
# 
# For more information check out the docstring of the `setup` function.


# help(setup)


# ## ✅ Create Model
# This function trains and evaluates the performance of a given estimator using cross-validation. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function. All the available models can be accessed using the models function.


# check all the available models
models()


# train kmeans
kmeans = create_model('kmeans')


# The function above has return trained model object as an output. The scoring grid is only displayed and not returned. If you need access to the scoring grid you can use `pull` function to access the dataframe.


kmeans_results = pull()
print(type(kmeans_results))
kmeans_results


# train kmeans with 10 clusters
create_model('kmeans', num_clusters = 10)


# Some other parameters that you might find very useful in `create_model` are:
# 
# - num_clusters
# - ground_truth
# - fit_kwargs
# - experiment_custom_tags
# - engine
# 
# You can check the docstring of the function for more info.


# help(create_model)


# ## ✅ Assign Model
# This function assigns cluster labels to the training data, given a trained model.


assign_model(kmeans)


# ## ✅ Plot Model
# This function analyzes the performance of a trained model.


# to control the scale of plot
plot_model(kmeans, plot = 'elbow', scale = 2)


# to save the plot
plot_model(kmeans, plot = 'elbow', save=True)


# Some other parameters that you might find very useful in `plot_model` are:
# 
# - feature
# - label
# - display_format
# 
# You can check the docstring of the function for more info.


# help(plot_model)


# ## ✅ Deploy Model
# This function deploys the entire ML pipeline on the cloud.
# 
# **AWS:**  When deploying model on AWS S3, environment variables must be configured using the command-line interface. To configure AWS environment variables, type `aws configure` in terminal. The following information is required which can be generated using the Identity and Access Management (IAM) portal of your amazon console account:
# 
# - AWS Access Key ID
# - AWS Secret Key Access
# - Default Region Name (can be seen under Global settings on your AWS console)
# - Default output format (must be left blank)
# 
# **GCP:** To deploy a model on Google Cloud Platform ('gcp'), the project must be created using the command-line or GCP console. Once the project is created, you must create a service account and download the service account key as a JSON file to set environment variables in your local environment. Learn more about it: https://cloud.google.com/docs/authentication/production
# 
# **Azure:** To deploy a model on Microsoft Azure ('azure'), environment variables for the connection string must be set in your local environment. Go to settings of storage account on Azure portal to access the connection string required.
# AZURE_STORAGE_CONNECTION_STRING (required as environment variable)
# Learn more about it: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


# deploy model on aws s3
# deploy_model(kmeans, model_name = 'my_first_platform_on_aws',
#             platform = 'aws', authentication = {'bucket' : 'pycaret-test'})


# load model from aws s3
# loaded_from_aws = load_model(model_name = 'my_first_platform_on_aws', platform = 'aws',
#                              authentication = {'bucket' : 'pycaret-test'})

# loaded_from_aws


# ## ✅ Save / Load Model
# This function saves the transformation pipeline and a trained model object into the current working directory as a pickle file for later use.


# save model
save_model(kmeans, 'my_first_model')


# load model
loaded_from_disk = load_model('my_first_model')
loaded_from_disk


# ## ✅ Save / Load Experiment
# This function saves all the experiment variables on disk, allowing to later resume without rerunning the setup function.


# save experiment
save_experiment('my_experiment')


# load experiment from disk
exp_from_disk = load_experiment('my_experiment', data=data)





