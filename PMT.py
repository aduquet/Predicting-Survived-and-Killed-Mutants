import re
import glob as gl
import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import mglearn as mglearn
import warnings
import pandas as pd
from pandas import Series
from pandas.plotting import scatter_matrix
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, \
    classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation  # For clustering
from sklearn.mixture import GaussianMixture  # For GMM clustering

warnings.filterwarnings('always')
# Font size and style
plt.rc('text', usetex=False)
plt.rcParams.update({'font.size': 12})
plt.rc('font', family='serif')


def encoder_c(data):
    # print('** Encoding Categorical Features of C program **')
    le = sklearn.preprocessing.LabelEncoder()

    typeOfMutant_data = data['typeOfMutant']
    Mutation_data = data['Mutation']
    typeReturn_data = data['typeReturn']
    location_data = data['location']

    typeOfMutant_data_encoded = le.fit_transform(typeOfMutant_data)
    Mutation_data_encoded = le.fit_transform(Mutation_data)
    typeReturn_data_encoded = le.fit_transform(typeReturn_data)
    location_data_encoded = le.fit_transform(location_data)

    for index, row in data.iterrows():
        if row['result'] == 'killed':
            data.at[index, 'result'] = 1
        else:
            data.at[index, 'result'] = 0

        data.at[index, 'typeOfMutant'] = typeOfMutant_data_encoded[index]
        data.at[index, 'Mutation'] = Mutation_data_encoded[index]
        data.at[index, 'typeReturn'] = typeReturn_data_encoded[index]
        data.at[index, 'location'] = location_data_encoded[index]

    target = data['result']
    data = data.drop(['result'], axis=1)

    return data, target


def encoder(df_train, df_test, v):
    if v == 0:
        # print('** Encoding Categorical Features of Cross-version scenario **')
        le = sklearn.preprocessing.LabelEncoder()

        operator_train = df_train['operator']
        methodReturn_train = df_train['methodReturn']
        operator_encoded_train = le.fit_transform(operator_train)
        methodReturn_encoded_train = le.fit_transform(methodReturn_train)

        operator_test = df_test['operator']
        methodReturn_test = df_test['methodReturn']
        operator_encoded_test = le.fit_transform(operator_test)
        methodReturn_encoded_test = le.fit_transform(methodReturn_test)

        for index, row in df_train.iterrows():
            if row['isKilled'] == 'yes':
                df_train.at[index, 'isKilled'] = 1
            else:
                df_train.at[index, 'isKilled'] = 0

            df_train.at[index, 'operator'] = operator_encoded_train[index]
            df_train.at[index, 'methodReturn'] = methodReturn_encoded_train[index]

        for index, row in df_test.iterrows():
            if row['isKilled'] == 'yes':
                df_test.at[index, 'isKilled'] = 1
            else:
                df_test.at[index, 'isKilled'] = 0

            df_test.at[index, 'operator'] = operator_encoded_test[index]
            df_test.at[index, 'methodReturn'] = methodReturn_encoded_test[index]

        data_train = df_train.drop(['isKilled'], axis=1)
        target_train = df_train['isKilled']

        data_test = df_test.drop(['isKilled'], axis=1)
        target_test = df_test['isKilled']

        return data_train, target_train, data_test, target_test

    if v == 1:
        # print('** Encoding Categorical Features of Cross-Project scenario**')
        le = sklearn.preprocessing.LabelEncoder()

        operator = df_train['operator']
        methodReturn = df_train['methodReturn']

        operator_encoded = le.fit_transform(operator)
        methodReturn_encoded = le.fit_transform(methodReturn)

        for index, row in df_train.iterrows():
            if row['isKilled'] == 'yes':
                df_train.at[index, 'isKilled'] = 1
            else:
                df_train.at[index, 'isKilled'] = 0

            df_train.at[index, 'operator'] = operator_encoded[index]
            df_train.at[index, 'methodReturn'] = methodReturn_encoded[index]

        data = df_train.drop(['isKilled'], axis=1)
        target = df_train['isKilled']

        return data, target


def calculateMutationScore(test_labels, predicted_labels):
    test_MutantCount = collections.Counter(test_labels)
    predicted_MutantCount = collections.Counter(predicted_labels)
    mutationScore = test_MutantCount[1] / len(test_labels)
    predictedMutationScore = predicted_MutantCount[1] / len(predicted_labels)
    return abs(predictedMutationScore - mutationScore)


def performanceMetrics(ML, subject, trainVersion, testVersion, vtr, vts, y_tst, prediction, v):
    warnings.filterwarnings('always')
    # print("\nTest set score: {:.3f}".format(np.mean(prediction == y_test)))
    # print("Test set accuracy: {:.3f}".format(forest.score(x_train, y)))
    # print('** Evaluating the Model**')

    accu = accuracy_score(y_tst, prediction)
    prec = precision_score(y_tst, prediction)

    Recall = recall_score(y_tst, prediction, average='weighted')
    fmeasure = f1_score(y_tst, prediction, average='weighted')
    auc = roc_auc_score(y_tst, prediction, average='weighted')
    warnings.filterwarnings('always')
    if v == 1:
        result_cache = {'ML_Algorithm': ML, 'subject': subject, 'trainVersion': trainVersion, 'vtr': vtr,
                        'testVersion': testVersion, 'vts': vts, 'Accuracy': accu, 'Precision': prec, 'Recall': Recall,
                        'FMeasure': fmeasure, 'AUC': auc,
                        'abs(ps-as)': (calculateMutationScore(y_tst, prediction))}

        return result_cache

    if v == 0:
        result_cache = {'ML_Algorithm': ML, 'subject': subject, 'Train': trainVersion, 'Test': testVersion,
                        'Accuracy': accu, 'Precision': prec, 'Recall': Recall,
                        'FMeasure': fmeasure, 'AUC': auc,
                        'abs(ps-as)': (calculateMutationScore(y_tst, prediction))}

        return result_cache


def RandomForestModel(x_train, y_train, x_test, y_test, fi):
    warnings.filterwarnings('always')
    # print('** Creating Random Forest Model ** ')
    ML = 'RandomForest'
    y = y_train.values.tolist()
    # forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #  max_depth=2, max_features='auto', max_leaf_nodes=None,
    # min_impurity_decrease=0.0, min_impurity_split=None,
    # min_samples_leaf=1, min_samples_split=0.003,
    # min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
    # oob_score=False, random_state=42, verbose=0, warm_start=False)
    # forest = RandomForestClassifier(n_estimators=100, random_state=1)
    forest = RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=100, random_state=1,
                                    min_samples_split=0.000001)
    forest.fit(x_train, y)
    y_tst = y_test.values.tolist()
    prediction = forest.predict(x_test)
    if fi == 1:
        importance, indice = feature_importance(forest)
        # print('\n Classification Report: \n', classification_report(y_tst, prediction))
        return ML, y_tst, prediction, importance, indice
    else:
        return ML, y_tst, prediction


def feature_importance(forest):
    warnings.filterwarnings('always')
    importance = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    # Print the feature ranking
    indices = np.argsort(importance)[::-1]
    indices = np.sort(indices)
    return importance, indices


def naivebayes(x_train, y_train, y_test, x_test):
    warnings.filterwarnings('always')
    ML = 'NaiveBayes'
    gnb = GaussianNB()
    y_tst = y_test.values.tolist()
    model = gnb.fit(x_train, y_train)
    prediction = model.predict(y_tst)
    return ML, y_tst, prediction


def svm(x_train, y_train, x_test, y_test):
    warnings.filterwarnings('always')
    # print('** Creating SVM Model **')
    ML = 'svm'
    y = y_train.values.tolist()
    y_tst = y_test.values.tolist()

    # compute the minimum value per feature on the training set
    min_on_training = x_train.min(axis=0)
    # compute the range of each feature (max - min) on the training set
    range_on_training = (x_train - min_on_training).max(axis=0)

    X_train_scaled = (x_train - min_on_training) / range_on_training
    x = X_train_scaled.fillna(X_train_scaled.mean(), inplace=True)
    svc = SVC(gamma='auto')
    svc.fit(x_train, y)
    prediction = svc.predict(x_test)
    return ML, y_tst, prediction


def save_result_cache(df, file_out):
    warnings.filterwarnings('always')
    """
    Function that stores the dataframe
    :param df: dataframe to be stored as CSV
    :param file_out: the name of the CSV file
    """
    file_out += '.csv'
    # write into CSV file
    df.to_csv(file_out)


def id_names_test(test_path):
    name_test = test_path.split('/')
    name_test = (name_test[-1])
    name_test = name_test.split('.')
    name_test = str(name_test[0])
    name_test = name_test.split('_')
    proj_version_test = name_test[0]
    proj_subjec_test = name_test[1]
    proj_name_version_test = name_test[2]

    return proj_version_test, proj_subjec_test, proj_name_version_test


def id_names_train(train_path):
    name_train = train_path.split('/')
    name_train = (name_train[-1])
    name_train = name_train.split('.')
    name_train = str(name_train[0])
    name_train = name_train.split('_')
    proj_version_train = name_train[0]
    proj_subjec_train = name_train[1]
    proj_name_version_train = name_train[2]

    return proj_version_train, proj_subjec_train, proj_name_version_train


def extractName(data_path):
    name = data_path.split('/')
    name = name[-1]
    name = name.split('.')
    name = str(name[0])
    return name


def save_csv(file_out, df_result_cache, df_feature_importance, title, sc):
    if sc == 0:
        df_result_cache.to_csv(file_out + '_' + title + '.csv')
        df_feature_importance.to_csv(file_out + '_' + title + 'FeatureImportance.csv')
        print('*** Saved ***')

    if sc == 1:
        print('despues lo uso')


def commonFeature_p(data, target, name, file_out, l):
    if l == 0:  # JAVA language
        title = 'Cross-project_commonFeatures'
        # print(data.dtypes)
        testSize = np.arange(0.5, 1, 0.1)
        data = data.drop(['DepthTree'], axis=1)
        data = data.drop(['NumSubclass'], axis=1)
        # data = data.drop(['McCabe'], axis=1)
        # data = data.drop(['LOC'], axis=1)
        data = data.drop(['DepthNested'], axis=1)
        data = data.drop(['CA'], axis=1)
        data = data.drop(['CE'], axis=1)
        data = data.drop(['Instability'], axis=1)
        data = data.drop(['numCovered'], axis=1)
        # data = data.drop(['operator'], axis=1)
        # data = data.drop(['methodReturn'], axis=1)
        # data = data.drop(['numTestsCover'], axis=1)
        # data = data.drop(['mutantAssert'], axis=1)
        # data = data.drop(['classAssert'], axis=1)

        for k in range(0, len(testSize)):
            ts = (np.round(testSize[k], 2))
            xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size=ts, random_state=0)
            ML, y_tst, prediction, importance_common, indices = RandomForestModel(xTrain, yTrain,
                                                                                  xTest, yTest, fi=1)
            importance_common = np.append(importance_common, name)
            importance_common = np.append(importance_common, ts)
            importance_common = np.append(importance_common, 1 - ts)
            fi_cache_common_f_list.append(importance_common)
            a = 0
            r = 0
            result_cache_common_f = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
            # Performance Metrics for Random Forest
            result_cache_common_f_list.append(result_cache_common_f)
            # print(result_cache)

            # SVM
            ML, y_tst, prediction = svm(xTrain, yTrain, xTest, yTest)
            # Performance Metrics for SVM
            result_cache_common_f = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
            result_cache_common_f_list.append(result_cache_common_f)
            # print(result_cache)

        head = data.columns.values
        # print(head)
        a = np.append(head, 'subject')
        a = np.append(a, 'Train')
        a = np.append(a, 'Test')
        # print(fi_cache_list)
        df_feature_importance_common = pd.DataFrame(fi_cache_common_f_list)
        # print(df_feature_importance)
        # print(a)
        df_feature_importance_common.columns = a
        # print('\n', df_feature_importance)
        df_result_cache_common = pd.DataFrame(result_cache_common_f_list)
        # print('\n', df_result_cache)

        # SAVE DATA IN A CVS FILE
        save_csv(file_out, df_result_cache_common, df_feature_importance_common, title, sc=0)

    if l == 1:
        title = 'C-project_commonFeatures'
        # print(data.dtypes)
        testSize = np.arange(0.5, 1, 0.1)
        data = data.drop(['MutantId'], axis=1)
        # data = data.drop(['typeOfMutant'], axis=1)
        data = data.drop(['Mutation'], axis=1)
        data = data.drop(['cfileId'], axis=1)
        data = data.drop(['methodId'], axis=1)
        data = data.drop(['Line'], axis=1)
        data = data.drop(['Column'], axis=1)
        data = data.drop(['lineInMethod'], axis=1)
        # data = data.drop(['numTestCovered'], axis=1)
        # data = data.drop(['numMutationAssertion_iparam'], axis=1)
        # data = data.drop(['numMutationAssertion_oparam'], axis=1)
        # data = data.drop(['numClassAssertions'], axis=1)
        # data = data.drop(['typeReturn'], axis=1)
        data = data.drop(['branches'], axis=1)
        data = data.drop(['loops'], axis=1)
        data = data.drop(['maintainability'], axis=1)
        # data = data.drop(['mccabe'], axis=1)
        # data = data.drop(['sloc'], axis=1)
        # data = data.drop(['lines'], axis=1)
        data = data.drop(['operands'], axis=1)
        data = data.drop(['operators'], axis=1)
        data = data.drop(['unique_operands'], axis=1)
        data = data.drop(['unique_operators'], axis=1)
        data = data.drop(['volume'], axis=1)
        data = data.drop(['location'], axis=1)

        for k in range(0, len(testSize)):
            ts = (np.round(testSize[k], 2))
            xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size=ts, random_state=0)
            ML, y_tst, prediction, importance_common, indices = RandomForestModel(xTrain, yTrain,
                                                                                  xTest, yTest, fi=1)
            importance_common = np.append(importance_common, name)
            importance_common = np.append(importance_common, ts)
            importance_common = np.append(importance_common, 1 - ts)
            fi_cache_common_f_list.append(importance_common)
            a = 0
            r = 0
            result_cache_common_f = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
            # Performance Metrics for Random Forest
            result_cache_common_f_list.append(result_cache_common_f)
            # print(result_cache)

            # SVM
            ML, y_tst, prediction = svm(xTrain, yTrain, xTest, yTest)
            # Performance Metrics for SVM
            result_cache_common_f = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
            result_cache_common_f_list.append(result_cache_common_f)
            # print(result_cache)

        head = data.columns.values
        # print(head)
        a = np.append(head, 'subject')
        a = np.append(a, 'Train')
        a = np.append(a, 'Test')
        # print(fi_cache_list)
        df_feature_importance_common = pd.DataFrame(fi_cache_common_f_list)
        # print(df_feature_importance)
        # print(a)
        df_feature_importance_common.columns = a
        # print('\n', df_feature_importance)
        df_result_cache_common = pd.DataFrame(result_cache_common_f_list)
        # print('\n', df_result_cache)

        # SAVE DATA IN A CVS FILE
        save_csv(file_out, df_result_cache_common, df_feature_importance_common, title, sc=0)


def commonFeature_v(data):
    data = data.drop(['DepthTree'], axis=1)
    data = data.drop(['NumSubclass'], axis=1)
    # data = data.drop(['McCabe'], axis=1)
    # data = data.drop(['LOC'], axis=1)
    data = data.drop(['DepthNested'], axis=1)
    data = data.drop(['CA'], axis=1)
    data = data.drop(['CE'], axis=1)
    data = data.drop(['Instability'], axis=1)
    data = data.drop(['numCovered'], axis=1)
    # data = data.drop(['operator'], axis=1)
    # data = data.drop(['methodReturn'], axis=1)
    # data = data.drop(['numTestsCover'], axis=1)
    # data = data.drop(['mutantAssert'], axis=1)
    # data = data.drop(['classAssert'], axis=1)
    return data


if __name__ == '__main__':

    import click

    global result_cache, fi_cache
    result_cache = collections.defaultdict(dict)
    result_cache_v0 = collections.defaultdict(dict)

    result_cache_common_f = collections.defaultdict(dict)
    result_cache_common_f_v0 = collections.defaultdict(dict)

    fi_cache = collections.defaultdict(dict)
    fi_cache_common_f = collections.defaultdict(dict)

    global result_cache_list, fi_cache_list
    result_cache_list = []
    result_cache_list_v0 = []

    result_cache_common_f_list = []
    result_cache_common_f_list_v0 = []

    fi_cache_common_f_list = []
    fi_cache_list = []


    @click.command()
    @click.option('-sc', '--version_scenario', 'v', help=' p for Cross-project, and v for Cross-version')
    @click.option('-i', '--file', 'file_in', help='the path for getting the data set')
    @click.option('-o', '--out', 'file_out', help='Name of the file in which the data will be stored.')
    @click.option('-l', '--language', 'pro_lang', help='j for Java languages and c for C language')
    def main(v, file_in, file_out, pro_lang):
        warnings.filterwarnings('always')
        global data_train, target_train_first_version, data_train_first_version, proj_subjec_train_first, proj_name_version_train_first, proj_version_train_first, df_result_cache, df_feature_importance, data_train_commonFeatures, data_train_first_version_commonfeatures

        if v == 'p':

            if pro_lang == 'j':
                title = 'Cross-Project'
                print('\n *** Cross-project scenario ***\n')
                data_path = gl.glob(file_in + '*')
                data_path.sort()
                # print(data_path2)
                df_empty = pd.DataFrame()
                for i in range(0, len(data_path)):
                    print('\n*** Reading Data ***')
                    data = pd.read_csv(data_path[i])
                    name = extractName(data_path[i])
                    data, target = encoder(data, df_empty, v=1)
                    commonFeature_p(data, target, name, file_out, l=0)
                    testSize = np.arange(0.5, 1, 0.1)

                    for k in range(0, len(testSize)):
                        ts = (np.round(testSize[k], 2))
                        xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size=ts, random_state=0)
                        ML, y_tst, prediction, importance, indices = RandomForestModel(xTrain, yTrain,
                                                                                       xTest, yTest, fi=1)
                        importance = np.append(importance, name)
                        importance = np.append(importance, ts)
                        importance = np.append(importance, 1 - ts)
                        fi_cache_list.append(importance)
                        a = 0
                        r = 0
                        result_cache = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
                        # Performance Metrics for Random Forest
                        result_cache_list.append(result_cache)
                        # print(result_cache)

                        # SVM
                        ML, y_tst, prediction = svm(xTrain, yTrain, xTest, yTest)
                        # Performance Metrics for SVM
                        result_cache = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
                        result_cache_list.append(result_cache)
                        # print(result_cache)
                    head = data.columns.values
                    a = np.append(head, 'subject')
                    a = np.append(a, 'Train')
                    a = np.append(a, 'Test')
                    # print(fi_cache_list)
                    df_feature_importance = pd.DataFrame(fi_cache_list)
                    df_feature_importance.columns = a
                    # print('\n', df_feature_importance)
                    df_result_cache = pd.DataFrame(result_cache_list)
                    # print('\n', df_result_cache)

                    # SAVE DATA IN A CVS FILE
                    save_csv(file_out, df_result_cache, df_feature_importance, title, sc=0)

            if pro_lang == 'c':
                title = 'C-Project'
                print('\n *** C-project ***\n')
                data_path = gl.glob(file_in + '*')
                data_path.sort()
                # print(data_path)
                df_empty = pd.DataFrame()
                for i in range(0, len(data_path)):
                    print('\n*** Reading Data ***')
                    data = pd.read_csv(data_path[i])
                    name = extractName(data_path[i])
                    # print(data.dtypes)
                    data, target = encoder_c(data)
                    testSize = np.arange(0.5, 1, 0.1)
                    commonFeature_p(data, target, name, file_out, l=1)

                    for k in range(0, len(testSize)):
                        ts = (np.round(testSize[k], 2))
                        xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size=ts, random_state=0)
                        ML, y_tst, prediction, importance, indices = RandomForestModel(xTrain, yTrain,
                                                                                       xTest, yTest, fi=1)
                        importance = np.append(importance, name)
                        importance = np.append(importance, ts)
                        importance = np.append(importance, 1 - ts)
                        fi_cache_list.append(importance)
                        a = 0
                        r = 0
                        result_cache = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
                        # Performance Metrics for Random Forest
                        result_cache_list.append(result_cache)
                        # print(result_cache)

                        # SVM
                        ML, y_tst, prediction = svm(xTrain, yTrain, xTest, yTest)
                        # Performance Metrics for SVM
                        result_cache = performanceMetrics(ML, name, ts, 1 - ts, a, r, y_tst, prediction, v=0)
                        result_cache_list.append(result_cache)
                        print(result_cache)
                    head = data.columns.values
                    a = np.append(head, 'subject')
                    a = np.append(a, 'Train')
                    a = np.append(a, 'Test')
                    # print(fi_cache_list)
                    df_feature_importance = pd.DataFrame(fi_cache_list)
                    df_feature_importance.columns = a
                    # print('\n', df_feature_importance)
                    df_result_cache = pd.DataFrame(result_cache_list)
                    # print('\n', df_result_cache)

                    # SAVE DATA IN A CVS FILE
                    save_csv(file_out, df_result_cache, df_feature_importance, title, sc=0)

        else:

            if pro_lang == 'j':
                title = 'Cross-Version'
                print('\n *** Cross-version scenario ***\n')
                # data_path = gl.glob(file_in + 'lang/v*')
                # data_path.sort()
                data_path2 = gl.glob(file_in + '*')
                data_path2.sort()
                # print(data_path2)
                for k in range(0, len(data_path2)):
                    data_path = gl.glob(data_path2[k] + '/v*')
                    data_path.sort()
                    # print(dataPath)
                    for i in range(0, len(data_path)):
                        # print("** Reading Data **")
                        if i != (len(data_path) - 1):
                            df_train = pd.read_csv(data_path[i])
                            df_test = pd.read_csv(data_path[i + 1])

                            proj_version_train, proj_subjec_train, proj_name_version_train = id_names_train(
                                data_path[i])
                            proj_version_test, proj_subjec_test, proj_name_version_test = id_names_test(
                                data_path[i + 1])

                            print('\n - Training data: ', proj_subjec_train, proj_version_train,
                                  proj_name_version_train)
                            print(' - Test data: ', proj_subjec_test, proj_version_test, proj_name_version_test, '\n')

                            # print("\nTrain data shape: ", df_train.shape)
                            # print("Test data shape: ", df_test.shape, '\n')
                            # print(df_train.select_dtypes(include=['object']))
                            data_train, target_train, data_test, target_test = encoder(df_train, df_test, v=0)

                            data_train_commonFeatures = commonFeature_v(data_train)
                            data_test_commonFeatures = commonFeature_v(data_test)

                            if i == 0:
                                data_train_first_version = data_train.copy()
                                target_train_first_version = target_train.copy()
                                proj_subjec_train_first = proj_subjec_train
                                proj_name_version_train_first = proj_name_version_train
                                proj_version_train_first = proj_version_train

                                data_train_first_version_commonfeatures = data_train_commonFeatures.copy()

                            # print("Train data shape: ", data_train.shape)
                            # print("Test data shape: ", data_test.shape)

                            # RANDOM FOREST
                            ML, y_tst, prediction, importance, indices = RandomForestModel(data_train, target_train,
                                                                                           data_test, target_test, fi=1)
                            importance = np.append(importance, proj_name_version_train)
                            importance = np.append(importance, proj_version_train)
                            importance = np.append(importance, proj_subjec_train)
                            fi_cache_list.append(importance)
                            # Performance Metrics for Random Forest
                            result_cache = performanceMetrics(ML, proj_subjec_test, proj_name_version_train,
                                                              proj_name_version_test, proj_version_train,
                                                              proj_version_test,
                                                              y_tst, prediction, v=1)
                            result_cache_list.append(result_cache)

                            # SVM
                            ML, y_tst, prediction = svm(data_train, target_train, data_test, target_test)
                            # Performance Metrics for SVM
                            result_cache = performanceMetrics(ML, proj_subjec_test, proj_name_version_train,
                                                              proj_name_version_test, proj_version_train,
                                                              proj_version_test,
                                                              y_tst, prediction, v=1)
                            result_cache_list.append(result_cache)

                            # -------------- TRAINING WITH THE FIRST VERSION -----------------------
                            ML, y_tst, prediction = RandomForestModel(data_train_first_version,
                                                                      target_train_first_version,
                                                                      data_test, target_test, fi=0)
                            result_cache_v0 = performanceMetrics(ML, proj_subjec_train_first,
                                                                 proj_name_version_train_first,
                                                                 proj_name_version_test, proj_version_train_first,
                                                                 proj_version_test, y_tst, prediction, v=1)
                            result_cache_list_v0.append(result_cache_v0)

                            ML, y_tst, prediction = RandomForestModel(data_train_first_version_commonfeatures,
                                                                      target_train_first_version,
                                                                      data_test_commonFeatures, target_test, fi=0)
                            result_cache_common_f_v0 = performanceMetrics(ML, proj_subjec_train_first,
                                                                          proj_name_version_train_first,
                                                                          proj_name_version_test,
                                                                          proj_version_train_first,
                                                                          proj_version_test, y_tst, prediction, v=1)
                            result_cache_common_f_list_v0.append(result_cache_common_f_v0)

                            ML, y_tst, prediction = svm(data_train_first_version_commonfeatures,
                                                        target_train_first_version, data_test_commonFeatures,
                                                        target_test)
                            # Performance Metrics for SVM
                            result_cache_common_f_v0 = performanceMetrics(ML, proj_subjec_train_first,
                                                                          proj_name_version_train_first,
                                                                          proj_name_version_test,
                                                                          proj_version_train_first,
                                                                          proj_version_test,
                                                                          y_tst, prediction, v=1)
                            result_cache_common_f_list_v0.append(result_cache_common_f_v0)

                            # ------------- Commun Features -----------------
                            ML, y_tst, prediction, importance_c, indices = RandomForestModel(data_train_commonFeatures,
                                                                                             target_train,
                                                                                             data_test_commonFeatures,
                                                                                             target_test, fi=1)
                            importance_c = np.append(importance_c, proj_name_version_train)
                            importance_c = np.append(importance_c, proj_version_train)
                            importance_c = np.append(importance_c, proj_subjec_train)
                            fi_cache_common_f_list.append(importance_c)
                            # Performance Metrics for Random Forest
                            result_cache_common_f = performanceMetrics(ML, proj_subjec_test, proj_name_version_train,
                                                                       proj_name_version_test, proj_version_train,
                                                                       proj_version_test,
                                                                       y_tst, prediction, v=1)
                            result_cache_common_f_list.append(result_cache_common_f)

                            # SVM
                            ML, y_tst, prediction = svm(data_train_commonFeatures, target_train,
                                                        data_test_commonFeatures, target_test)
                            # Performance Metrics for SVM
                            result_cache_common_f = performanceMetrics(ML, proj_subjec_test, proj_name_version_train,
                                                                       proj_name_version_test, proj_version_train,
                                                                       proj_version_test,
                                                                       y_tst, prediction, v=1)
                            result_cache_common_f_list.append(result_cache_common_f)

                    head = data_train.columns.values
                    head2 = data_train_commonFeatures.columns.values
                    a = np.append(head, 'Version_name')
                    a = np.append(a, 'Version')
                    a = np.append(a, 'subject')
                    a2 = np.append(head2, 'Version_name')
                    a2 = np.append(a2, 'Version')
                    a2 = np.append(a2, 'subject')
                    # print(fi_cache_list)
                    df_feature_importance = pd.DataFrame(fi_cache_list)
                    df_feature_importance.columns = a

                    df_feature_importance_commonFeatures = pd.DataFrame(fi_cache_common_f_list)
                    df_feature_importance_commonFeatures.columns = a2

                    df_result_cache_common = pd.DataFrame(result_cache_common_f_list)
                    df_result_cache = pd.DataFrame(result_cache_list)
                    df_result_cache_first_version = pd.DataFrame(result_cache_list_v0)
                    df_result_cache_common_first_version = pd.DataFrame(result_cache_common_f_list_v0)
                    # print(df_result_cache_first_version)
                    # print(df_result_cache)

                    # SAVE DATA IN A CVS FILE
                    save_csv(file_out, df_result_cache, df_feature_importance, title, sc=0)
                    df_result_cache_first_version.to_csv(file_out + '_' + title + '_Training_firstVersion.csv')
                    df_result_cache_common.to_csv(file_out + '_' + title + '_commonFeatures.csv')
                    df_feature_importance_commonFeatures.to_csv(
                        file_out + '_' + title + '_commonFeaturesImportance.csv')
                    df_result_cache_common_first_version.to_csv(
                        file_out + '_' + title + 'Training_firstVersion_commonFeatures.csv')


    main()
