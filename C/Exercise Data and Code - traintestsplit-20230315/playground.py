import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

# %%
# read the data ----------------------
# -- pre coded --
pth = 'data_ex1ex3_waterquality.csv'
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced! (attribute is_safe = y)
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
# -- student work --

y = df['is_safe'].to_numpy()
df.drop(['is_safe'], inplace=True, axis=1)
X = df.to_numpy()

# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X = mmSc.fit_transform(X)

# perform holdout validation without shuffle and stratify -------
# use k settings 1,2,3,...7
# -- student work --

# split without shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=False)
results_without_shuffle = []

for i in range(1, 8):
    _i_nn_cls = KNeighborsClassifier(i)

    _i_nn_cls = _i_nn_cls.fit(X_train, y_train)
    y_hat_classification_i = _i_nn_cls.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, y_hat_classification_i), 3)
    balanced_accuracy = np.round(balanced_accuracy_score(y_test, y_hat_classification_i), 3)
    # TODO roc_auc

    results_dict = {"k": i, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "roc_auc": None,
                    "shuffle": False}
    results_without_shuffle.append(results_dict)

# perform holdout validation WITH shuffle but NO stratify -----
# again, use k settings 1,2,3,...7
# -- student work --

# split without shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)
results_with_shuffle = []

for i in range(1, 8):
    _i_nn_cls = KNeighborsClassifier(i)

    _i_nn_cls = _i_nn_cls.fit(X_train, y_train)
    y_hat_classification_i = _i_nn_cls.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, y_hat_classification_i), 3)
    balanced_accuracy = np.round(balanced_accuracy_score(y_test, y_hat_classification_i), 3)
    # TODO roc_auc

    results_dict = {"k": i, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "roc_auc": None,
                    "shuffle": True}
    results_with_shuffle.append(results_dict)

# build the dataframe
# -- student work --
all_results = results_without_shuffle + results_with_shuffle
new_df = pd.DataFrame(all_results, columns=['k', 'accuracy', 'balanced_accuracy', 'roc_auc', 'shuffle'])