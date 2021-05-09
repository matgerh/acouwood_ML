import os
import pandas as pd
from math import sqrt
import mlflow
from scipy import fftpack
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics



mlflow.set_tracking_uri("Model selection")


# ML experiment name
mlflow.set_experiment("plot")

pd.set_option('display.max_rows', None)

with mlflow.start_run(run_name="SVM"): 

    ##############################################################################
    # Feature extraction

    # Compute features 
    def features(dft):
        rmm_features = [] # List to store features (RMS, maximum, and minimum)

        # Store column for easier access
        df_freq = dft['Frequency (Hz)'] # Frequencies
        df_sp = dft['Level (dB)']  # Sound pressure level

        RMS_all = compute_RMS(df_sp) # RMS for whole spectrum
        max_all = df_sp.max() # Maximum for whole spectrum
        min_all = df_sp.min() # Minimum for whole spectrum
        rmm_features.extend([RMS_all,max_all,min_all]) # Add list of features to overall list of features 

        # Seperate data based on 5 intervals and store in dataframes
        df_sp_r1 = df_sp[df_freq <= 4410] # Below or equal to 4410 Hz
        df_sp_r2 = df_sp[(4410 < df_freq) & (df_freq <= 8820)] # Above 4410 Hz and below or equal to 8820 Hz
        df_sp_r3 = df_sp[(8820 < df_freq) & (df_freq <= 13230)] # Above 8820 Hz and below or equal to 13230 Hz
        df_sp_r4 = df_sp[(13230 < df_freq) & (df_freq <= 17640)] # Above 13230 Hz and below or equal to 17640 Hz
        df_sp_r5 = df_sp[(17640 < df_freq)] # Above 17640 Hz

        # Store dataframe in list that can be iterated
        df_list = [df_sp_r1,df_sp_r2,df_sp_r3,df_sp_r4,df_sp_r5] 

        # For each interval
        for df in df_list:
            RMS = compute_RMS(df) # RMS of interval
            MAX = df.max() # Maximum of interval
            MIN = df.min() # Minimum of interval 
            rmm_features.extend([RMS,MAX,MIN]) # Add list of features to overall list of features 

        # Return a list of computed features 
        return rmm_features 

    # Calculating RMS of a data structure
    def compute_RMS(V):
        n = len(V) # Number of data points 
        return sqrt(sum(v*v for v in V)/n)  # Square all numbers, find the mean of them, and take the square root of the result

    # Fast fourier transform of data 
    def FFT(data,sr):
        N = 2048 # Number of data points used
        sp_values = np.fft.rfft(data, N) # Applying FFT to compute the sound pressure values of discrete Fourier Transform (DFT) 
        sp_mag = np.abs(sp_values) # Convert to magnitudes (absolute values of signal)
        sp_dB = np.abs(20*np.log10(sp_mag/max(sp_values))) # Convert magnitudes to decibels
        d = 1.0/sr # Sample spacing (inverse of sampling rate)
        freq = np.fft.rfftfreq(N,d=d) # Return the Discrete Fourier Transform sample frequencies
        dft = pd.DataFrame(freq,columns=["Frequency (Hz)"]) # Initiate dataframe with frequencies
        dft['Level (dB)'] = sp_dB # Add decibels to dataframe

        # print(len(sp_values))
        # plt.plot(freq, sp_dB)
        # plt.show()
        return dft

    # Directory of audio files
    dir = 'wav/'

    # List to store features 
    features_list = []

    # List to store classes
    classes = []

    # For each filename in folder 'wav'
    for filename in os.listdir(dir):
        c = filename[4:5] # Store the 5th character of file in c (category)
        if c == "d" or c == "e": # If category is either decay or extreme decay 
            continue # Then skip file 
        else:
            filename_path = os.path.join(dir, filename) # Create the path for the audio file
            data, sr = librosa.load(filename_path, sr=44100) # Read in audio file with sample rate 44,100 Hz
            df_dft = FFT(data,sr) # Convert from frequency-domain to time-domain 
            features_list.append(features(df_dft)) # Compute features and add to features list 
            classes.append(c) # Append class to class list 

    # Feature names in order of calculation 
    columns_list = ['RMS_all', 'MAX_all','MIN_all','RMS_r1', 'MAX_r1,', 'MIN_r1',
                    'RMS_r2','MAX_r2,', 'MIN_r2','RMS_r3','MAX_r3,', 'MIN_r3',
                    'RMS_r4','MAX_r4,', 'MIN_r4','RMS_r5', 'MAX_r5,', 'MIN_r5'] 

    df_data = pd.DataFrame(features_list, columns = columns_list) # Create dataframe and store calculated features with feature names 
    df_data['class'] = classes # Add column with class labels to dataframe

    ##################################################################
    # Outlier detection 
    from sklearn.neighbors import LocalOutlierFactor

    # fig, ax = plt.subplots()

    # colors = {0:'red', 1:'green'}
    # ax.scatter(df_data['MIN_r5'], df_data['RMS_r5'], color=df_data['class'].map(colors))
    # plt.show()

    # Outlier detection model
    model = LocalOutlierFactor()
    y_pred = model.fit_predict(df_data[columns_list])

    outlier_index = np.where(y_pred == -1)  # Filter outlier index
    outlier_values = df_data.iloc[outlier_index] # Filter outlier values

    # Visualize outliers 
    # plt.scatter(df_data['MIN_r5'], df_data['RMS_r5'])
    # plt.scatter(outlier_values['MIN_r5'], outlier_values['RMS_r5'], color='r')
    # plt.show()

    # drop outlier values 
    df_data = df_data.drop(outlier_index[0])
    df_data = df_data.reset_index(drop=True)

    ########################################################################
    # Data exploration

    # # Feature exploration 
    # test = SelectKBest(score_func=f_classif, k='all')
    # fit = test.fit(X, y)
    # features = fit.transform(X)
    # # Print the scores for the features
    # for i in range(len(fit.scores_)):
    # 	print('Feature %d: %f' % (i, fit.scores_[i]))

    # # Plot the scores
    # plt.bar(columns_list, fit.scores_, color="#47903A")
    # ax = plt.gca()
    # ax.tick_params(axis='x', labelsize=6)
    # plt.title('Feature scores')
    # #plt.show()

    # # Distribution - Check the proportion of "n" and "m" observations in the target variable 
    # print(data['class'].value_counts()/len(data['class']))   
    # plt.hist(data['RMS_all'], bins=20)
    # plt.show()

   

    #####################################################################################
    # Import sklearn modules
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.compose import ColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
    from sklearn.metrics import classification_report
    from numpy import set_printoptions
    from sklearn.feature_selection import SelectKBest, f_classif

    # Encode non-numeric categorical target variable to numeric
    le = LabelEncoder() 
    df_data['class'] = le.fit_transform(df_data['class'])

    feature_names = columns_list # Define the name of features used
    class_names = ['class'] # Define name of target variable

    # Store data from df in X and y variables 
    X = df_data[feature_names]
    y = df_data[class_names]

    # Debug pipeline by printing X (features set)
    class Debug(BaseEstimator, TransformerMixin):
        def fit(self,X,y): return self
        def transform(self,X): 
            print(X)
            return X
    
    number_of_features = 6

    # Build pipeline
    pipeline = Pipeline([
                        ("columns", ColumnTransformer([
                            ("scaler", StandardScaler(), columns_list),
                            ], remainder="passthrough")),
                    ('f_classif', SelectKBest(f_classif, k=number_of_features)),
                    #("encoder", OneHotEncoder()),
                    #("debug", Debug()),   
                    #("PCA", PCA(n_components=8)),
                    ("model", RandomForestClassifier()),     
                ])

    # Splits in cross validation
    number_of_splits = 3

    # Metrics for model evaluation
    metrics = [("Accuracy", accuracy_score, []), ("Precision", precision_score, []), ("Recall", recall_score, []),]

    # Log parameters to ML Flow
    mlflow.log_param("no_of_features", number_of_features)
    mlflow.log_param("no_of_splits", number_of_splits)
    mlflow.log_param("transformers", pipeline["columns"])
    mlflow.log_param("model", pipeline["model"])

    # Cross validation 
    for train, test in StratifiedKFold(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train], y.iloc[train].values.ravel()) # Fit pipeline to data
        predictions = pipeline.predict(X.iloc[test]) # Make predictions
        truth = y.iloc[test] # Get the true values of y
                    
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            if (name == "Accuracy"):
                score = func(truth, predictions) # Score between true values and predictions of y
            else: 
                score = func(truth, predictions, pos_label=0)
            scores.append(score) # Add score to list

    # Log summary of the metrics
    for name, _, scores in metrics:
        mean_score = sum(scores)/number_of_splits # Mean of the scores
        std_score = statistics.stdev(scores) # Variance (standard deviation) of the scores

        # # Log metrics to ML Flow
        mlflow.log_metric(f"mean_{name}", mean_score) 
        mlflow.log_metric(f"std_{name}", std_score)

        # Print scores
        print(f"mean_{name}", mean_score)  
