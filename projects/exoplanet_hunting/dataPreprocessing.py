import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


TRAIN_META = pd.read_csv('datasets/TRAIN_META.csv')
TEST_META = pd.read_csv('datasets/TEST_META.csv')


# Change classes from 1, 2 to 0, 1
def lableChange(y_train, y_test):
    labler = lambda x: 1 if x == 2 else 0
    y_train_01, y_test_01 = y_train.apply(labler), y_test.apply(labler)

    return y_train_01, y_test_01


# Define training and testing datasets
def datasets():

    x_train, y_train = TRAIN_META.loc[:, TRAIN_META.columns != 'LABEL'], TRAIN_META.loc[:, 'LABEL']
    x_test, y_test = TEST_META.loc[:, TEST_META.columns != 'LABEL'], TEST_META.loc[:, 'LABEL']

    #fill NaNs with mean (no NaNs)
    #for column in x_train:
        #x_train[column] = x_train[column].fillna(x_train[column].mean(), 2))

    #x_train, x_test = scale(x_train, x_test)
    #x_train = handle_outliers(x_train, 1)
    #print(sum(y_train > 1), sum(y_train < 2))
    x_train, y_train = smote(x_train, y_train)
    #print(sum(y_train > 1), sum(y_train < 2))
    y_train, y_test = lableChange(y_train, y_test)

    n_features = x_train.shape[1]

    return x_train, y_train, x_test, y_test, n_features


# Change value of outliers to mean
def handleOutliers(dataset, num_iterations):

    dataset_handled = dataset
    threshold = 100

    for n in range(num_iterations):

        # for column in range(dataset_handled.shape[0]):
        # for index, row in dataset_handled.iterrows():
            # row_values = row.values
            # row_max, row_min = row_values.max(), row_values.min()
            # row_maxidx, row_minidx = row_values.argmax(), row_values.argmin()
            # row_mean = row_values.mean()

            # #if np.abs(column_max/column_mean) >= threshold:
            # dataset_handled.iloc[index][row_maxidx] = row_mean

            # #if np.abs(column_min/column_mean) >= threshold:
            # dataset_handled.iloc[index][row_minidx] = row_mean

        for index, row in dataset_handled.iterrows():

            row_values = row.values
            #row_median = np.median(row_values)

            for i in range(1, len(row_values) - 1):

                prev_val = row_values[i - 1]
                val = row_values[i]
                next_val = row_values[i + 1]

                if abs(val) > threshold * abs(prev_val):
                    dataset_handled.iloc[index][i] = dataset_handled.iloc[index][i - 1]
                elif abs(val) > threshold * abs(next_val):
                    dataset_handled.iloc[index][i] = dataset_handled.iloc[index][i + 1]

    return dataset_handled


# Scale data
def scale(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


# SMOTE
def smote(x_train, y_train):

    #smote = SMOTE(random_state=17, sampling_strategy='minority')
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.3)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)

    return x_train_res, y_train_res


# Graph of data
def fluxGraph(dataset, row, planet):

    fig = plt.figure(figsize=(20,5), facecolor=(.18, .31, .31))
    ax = fig.add_subplot()
    ax.set_facecolor('#004d4d')
    ax.set_title(planet, color='white', fontsize=22)
    ax.set_xlabel('time', color='white', fontsize=18)
    ax.set_ylabel('flux_' + str(row), color='white', fontsize=18)
    ax.grid(False)
    flux_time = list(dataset.columns)
    flux_values = dataset[flux_time].iloc[row]
    ax.plot([i + 1 for i in range(dataset.shape[1])], flux_values, '#00ffff')
    ax.tick_params(colors = 'black', labelcolor='#00ffff', labelsize=14)
    plt.show()


# Show graph of data
def showGraph():

    with_planet = TRAIN_META[TRAIN_META['LABEL'] == 2].head(2).index
    wo_planet = TRAIN_META[TRAIN_META['LABEL'] == 1].head(2).index

    dataset, _, _, _, _ = datasets()
    time_points = list(dataset.columns)

    for row in with_planet:
        fluxGraph(dataset, row, planet = 'with_planet')
    for row in wo_planet:
        fluxGraph(dataset, row, planet = 'wo_planet')
