import random
import numpy as np
import pandas as pd
from copy import deepcopy as cp

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score
from imblearn.metrics import specificity_score

from keras.models import Sequential
from keras.layers import Dense, Dropout

import tensorflow as tf

random.seed(1032021)
np.random.seed(1032021)
tf.random.set_seed(1032021)


def general_grid_search(x, y, model, param, kfold):
    np.random.seed(1032021)

    grid = GridSearchCV(model,
                        param_grid=param,
                        verbose=2,
                        scoring="accuracy",
                        cv=kfold,
                        n_jobs=-1)
    grid.fit(x, y)

    return grid


def pca_lda_grid_search(x, y, kfold):
    pca = PCA()
    pca.fit(x)
    accuracy_outer = []

    for idx in range(1, x.shape[-1] + 1):
        accuracy_inner = []

        for train_index, val_index in kfold.split(x, y):
            x_train, x_val = x.iloc[train_index, :], x.iloc[val_index, :]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            pca = PCA()
            pca.fit(x_train)
            reduction_train = pca.transform(x_train)
            reduction_val = pca.transform(x_val)

            reduction_train, reduction_val = reduction_train[:, :idx], reduction_val[:, :idx]

            pca_lda = LinearDiscriminantAnalysis()
            pca_lda.fit(reduction_train, y_train)
            predictions = pca_lda.predict(reduction_val)

            accuracy_inner.append(accuracy_score(y_val, predictions))
        mean_inner = np.array(accuracy_inner).mean()
        accuracy_outer.append(mean_inner)

    accuracy_outer = np.array(accuracy_outer)
    print(np.argmax(accuracy_outer), accuracy_outer.max())


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def create_model(num_neurons, drop_out_rate, input_shape, output_shape=4):
    nn_model = Sequential()

    nn_model.add(Dense(num_neurons, activation='relu', input_shape=[input_shape[-1]]))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))
    nn_model.add(Dense(num_neurons, activation='relu'))
    nn_model.add(Dropout(drop_out_rate))

    nn_model.add(Dense(output_shape, activation='softmax'))

    nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn_model


def mlp_kfold(x, y, model, kfold, folder, epochs):
    count = 1
    accuracy = []
    for train_index, val_index in kfold.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        filepath = folder + get_model_name(count)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                        monitor='val_accuracy',
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        verbose=1)

        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[checkpoint])
        accuracy.append(history.history["val_accuracy"])
        count += 1

    return accuracy

def kfold_cross_validation(x, y, forest, svm, xgb, mlp, pca_lda, pca_idx, kfold, mlp_folder):
    count = 1
    kfold_results = {}
    label_encoder = LabelEncoder()
    evaluation = {'pca_lda': [], "svm": [], "forest": [], "xgb": [], 'mlp': [], 'y': []}
    for train_index, val_index in kfold.split(x, y):
        print("TRAIN:", train_index, "TEST:", val_index)

        models = {'pca_lda': [], "svm": [], "forest": [], "xgb": [], 'mlp': []}
        predictions = []

        x_train, x_val = x.iloc[train_index, :], x.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        y_num = label_encoder.fit_transform(y)
        y_train_num, y_val_num = y_num[train_index], y_num[val_index]

        pca = PCA()
        pca.fit(x_train)
        reduction_train = pca.transform(x_train)
        reduction_val = pca.transform(x_val)

        reduction_train, reduction_val = reduction_train[:, :pca_idx + 1], reduction_val[:, :pca_idx + 1]

        # RandomForestClassifier
        forest.fit(x_train, y_train)
        forest_prediction = forest.predict(x_val)

        # SVM
        svm.fit(x_train, y_train)
        svm_prediction = svm.predict(x_val)

        # XGBoost
        xgb.fit(x_train, y_train_num)
        xgb_prediction = xgb.predict(x_val)

        # MLP
        mlp.load_weights(mlp_folder + get_model_name(count))
        mlp_prediction = mlp.predict(x_val)
        mlp_prediction = np.argmax(mlp_prediction, axis=1)

        # PCA-LDA
        pca_lda.fit(reduction_train, y_train)
        pca_lda_prediction = pca_lda.predict(reduction_val)

        # predictions
        predictions.append(pca_lda_prediction)
        predictions.append(svm_prediction)
        predictions.append(forest_prediction)
        predictions.append(xgb_prediction)
        predictions.append(mlp_prediction)

        evaluation['pca_lda'].append(pca_lda_prediction)
        evaluation['svm'].append(svm_prediction)
        evaluation['forest'].append(forest_prediction)
        evaluation['xgb'].append(label_encoder.inverse_transform([round(label) for label in xgb_prediction]))
        evaluation['mlp'].append(label_encoder.inverse_transform([round(label) for label in mlp_prediction]))
        evaluation['y'].append(y_val)

        for idx, val in enumerate(models):
            if val == "xgb" or val == 'mlp':
                predictions[idx] = label_encoder.inverse_transform([round(label) for label in predictions[idx]])

            models[val] = classification_report(y_val, predictions[idx], output_dict=True)

        kfold_results[f'{count}'] = models

        count += 1

    return kfold_results, evaluation


def visualize_results(evaluation, y):
    models = {'pca_lda': [], "svm": [], "forest": [], "xgb": [], 'mlp': []}

    metrics = ["specificity", "sensitivity", "precision"]
    multi_index = []
    for model in models.keys():
        for metric in metrics:
            multi_index.append((model, metric))
    multi_index = pd.MultiIndex.from_tuples(multi_index)

    df = pd.DataFrame(columns=np.sort(y.unique()), index=multi_index)

    # %%
    temp = []
    final = {"specificity": [], "sensitivity": [], "precision": []}
    models = {'pca_lda': cp(final), "svm": cp(final), "forest": cp(final),
              "xgb": cp(final), 'mlp': cp(final)}
    flag = True

    for key in evaluation.keys():
        if key != "y":
            y_label = evaluation["y"]
            for i in range(15):
                test_true = y_label[i]
                test_pred = evaluation[key][i]
                models[key]["specificity"].append(specificity_score(test_true, test_pred, average=None))
                models[key]["sensitivity"].append(recall_score(test_true, test_pred, average=None))
                models[key]["precision"].append(precision_score(test_true, test_pred, average=None))

                if flag: temp.append(np.sort(test_true.unique()))
            flag = False

            for metric in metrics:
                df_temp = pd.DataFrame(index=range(len(temp)), columns=np.sort(y.unique()))
                for j in range(len(temp)):
                    df_temp.loc[j, temp[j]] = models[key][metric][j]

                    for label in df_temp.columns:
                        df_temp_label = df_temp.loc[:, label]
                        df_temp_mean = (df_temp_label.mean(axis=0) * 100).round(2)
                        df_temp_std = (df_temp_label.std(axis=0) * 100).round(2)
                        df.loc[(key, metric), label] = f"{df_temp_mean} +- {df_temp_std}"

    return df