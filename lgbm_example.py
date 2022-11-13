import time
from lightgbm import LGBMRegressor
import lightgbm as lgb
import gc

from numerapi import NumerAPI
from utils_plus import *

model_name = "lgbm_example"
model_path = ""
# f'models/LGBM/{model_name}.txt'
# Use this to load the model that you have saved

start = time.time()
napi = NumerAPI()
spinner = Halo(text='', spinner='dots')
current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

download_data()

# read the feature metadata and get the "small" feature set (We will need a data center grade server if we want to
# process all the Numerai features!)
with open("./data/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"]

# read the training and validation data given the predefined features stored in parquets as pandas DataFrames
training_data, validation_data = read_learning_data(features)
# extract feature matrix and target vector used for training
X_train = training_data.filter(like='feature_', axis='columns')
y_train = training_data[TARGET_COL]
# extract feature matrix and target vector used for validation
X_val = validation_data.filter(like='feature_', axis='columns')
y_val = validation_data[TARGET_COL]
# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()

########################################################################################################################
# define and train your model here using the loaded data sets!

num_feature_neutralization = 50  # parameter for feature neutralization used after we make our predictions
params = {"n_estimators": 5000,
          "learning_rate": 0.005,
          "max_depth": 10,
          "num_leaves": 2 ** 6,
          "colsample_bytree": 0.5,
          "subsample": 0.1,
          "device": "gpu",
          "verbosity": 1
          }

model = LGBMRegressor(**params)

if not model_path:
    spinner.start('Training model')
    model.fit(X_train, y_train)
    spinner.succeed()
    gc.collect()
    spinner.start('Saving model')
    model.booster_.save_model(f'models/LGBM/{model_name}.txt')
    spinner.succeed()
else:
    model = lgb.Booster(model_file=model_path)
    print(f'Model successfully read from {model_path}\n')

########################################################################################################################


spinner.start('Predicting on validation data')
# here we insert the predictions back into the validation data set, as we need to use the validation data set to
# perform feature neutralization later
validation_data.loc[:, f"preds_{model_name}"] = model.predict(X_val)
spinner.succeed()
gc.collect()

spinner.start('Neutralizing to risky features')
# neutralize our predictions to the k riskiest features in the training set
neutralize_riskiest_features(training_data, validation_data, features, model_name, k=num_feature_neutralization)
spinner.succeed()

gc.collect()

print('Exporting Predictions to csv...')
validation_data["prediction"] = validation_data[f"preds_{model_name}_neutral_riskiest_{num_feature_neutralization}"] \
    .rank(pct=True)
validation_data["prediction"].to_csv(f"predictions/LGBM/{model_name}.csv")
print('Done!')

spinner.start('Predicting on training data (making training data for meta-model)')
# here we insert the predictions back into the validation data set, as we need to use the validation data set to
# perform feature neutralization later
training_data.loc[:, f"preds_{model_name}"] = model.predict(X_train)
spinner.succeed()
gc.collect()

spinner.start('Neutralizing to risky features')
# neutralize our predictions to the k riskiest features in the training set
neutralize_riskiest_features(training_data, training_data, features, model_name, k=num_feature_neutralization)
spinner.succeed()
gc.collect()

print('Exporting model training data to csv...')
training_data["prediction"] = training_data[f"preds_{model_name}_neutral_riskiest_{num_feature_neutralization}"] \
    .rank(pct=True)
training_data["prediction"].to_csv(f"predictions/LGBM/meta_train_{model_name}.csv")
print('Done!')

print(f'Time elapsed: {(time.time() - start) / 60} mins')
