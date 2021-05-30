import os
import pandas as pd
import mlflow
import librosa
import matplotlib.pyplot as plt
import numpy as np
import statistics

mlflow.set_tracking_uri("http://localhost:5000/") # Traking uri
mlflow.set_experiment("Feauture_selection") # MLFLow experiment name

with mlflow.start_run(run_name="KNN"): # MLName of run

    ##############################################################################
    # Feature extraction

    # Fast fourier transform of data 
    def FFT(data,sr):
        N = 2048 # Number of data points used
        sp_values = np.fft.rfft(data, N) # Applying FFT to compute the sound pressure values in frequency-domain 
        sp_mag = np.abs(sp_values) # Convert to magnitudes (absolute values of signal)
        sp_dB = np.abs(20*np.log10(sp_mag/max(sp_values))) # Convert magnitudes to decibels
        d = 1.0/sr # Sample spacing (inverse of sampling rate)
        freq = np.fft.rfftfreq(N,d=d) # Frequencies of FFT
        df_fft = pd.DataFrame(freq,columns=["Frequency (Hz)"]) # Initiate dataframe with frequencies
        df_fft['Level (dB)'] = sp_dB # Add sound pressure level in decibels to dataframe

        # Return data frame with frequency-domain data
        return df_fft

    # Compute features in frequency intervals
    def interval_features(dft):
        interval_features = [] # List to store mean features 

        # Store column for easier access
        df_freq = dft['Frequency (Hz)'] # Frequencies
        df_sp = dft['Level (dB)']  # Sound pressure level

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
            Mean = df.mean() # Mean of interval
            Max = df.max() # Maximum of interval
            Min = df.min() # Minimum of interval
            interval_features.extend([Mean,Max,Min]) # Add list of features to overall list of features 

        # Return a list of computed features 
        return interval_features 

    # Compute mean spectral features 
    def spectral_features(y, sr):
        sp = librosa.feature.spectral_centroid(y=y, sr=sr)[0] # Compute spectral controids and store in array
        sp_mean = np.mean(sp) # Mean of spectral centroids
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0] # Compute spectral bandwidths and store in array
        sb_mean = np.mean(sb) # Mean of spectral bandwidth
        spectral_features = [sp_mean, sb_mean] # Store in list

        # Return a list of computed features
        return spectral_features 
        
    dir = 'wav/' # Directory of audio files
    features_list = [] # List to store features         
    classes = [] # List to store classes

    # For each filename in folder 'wav'
    for filename in os.listdir(dir):
        c = filename[4:5] # Store the 5th character of file in c (category)
        if c == "d" or c == "e": # If category is either decay or extreme decay 
            continue # Then skip file 
        else:
            filename_path = os.path.join(dir, filename) # Create the path for the audio file
            data, sr = librosa.load(filename_path, sr=44100) # Read in audio file with sample rate 44,100 Hz
            df_fft = FFT(data,sr) # Convert from frequency-domain to time-domain 

            i_features = interval_features(df_fft) # Compute interval features 
            s_features = spectral_features(data, sr) # Compute spectal features
            i_features.extend(s_features) # Combine features intro one list

            features_list.append(i_features) # Add to overall features list
            classes.append(c) # Append class to class list 

    # Feature names in order of calculation 
    columns_list = ['mean_r1', 'max_r1,', 'min_r1','mean_r2','max_r2', 'min_r2',
                    'mean_r3','max_r3', 'min_r3','mean_r4','max_r4', 'min_r4',
                    'mean_r5', 'max_r5,', 'min_r5','sp_mean','sb_mean'] 

    df_data = pd.DataFrame(features_list, columns = columns_list) # Create dataframe and store calculated features with feature names 
    df_data['class'] = classes # Add column with class labels to dataframe

    #####################################################################################
    # Analytics 
    
    # Import sklearn modules
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    # Encode non-numeric categorical target variable to numeric
    le = LabelEncoder() 
    df_data['class'] = le.fit_transform(df_data['class'])

    feature_names = columns_list # Define the name of features used
    class_names = ['class'] # Define name of target variable

    # Store data from df in X and y variables 
    X = df_data[feature_names]
    y = df_data[class_names]

    # Outlier detection model
    model = LocalOutlierFactor(n_neighbors=10)
    y_pred = model.fit_predict(df_data[columns_list])

    outlier_index = np.where(y_pred == -1)  # Filter outlier index
    outlier_values = df_data.iloc[outlier_index] # Filter outlier values

    df_data = df_data.drop(outlier_index[0]) # Drop outlier values 
    df_data = df_data.reset_index(drop=True) # Reset index 

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
                            ("scaler", MinMaxScaler(), columns_list),
                            ], remainder="passthrough")),
                    ('f_classif', SelectKBest(mutual_info_classif, k=number_of_features)),
                    #("debug", Debug()),   
                    ("model", KNeighborsClassifier()),     
                ])

    # Splits in cross validation
    number_of_splits = 3

    # Metrics for model evaluation
    metrics = [("Accuracy", accuracy_score, []), ("Precision", precision_score, []), ("Recall", recall_score, []),]

    # Log parameters to ML Flow
    mlflow.log_param("no_of_features", number_of_features) # Log number of input features
    mlflow.log_param("no_of_splits", number_of_splits) # Log number of split in cross validation
    mlflow.log_param("transformers", pipeline["columns"]) # Log transformers used for preprocessing
    mlflow.log_param("model", pipeline["model"]) # Log the ML model 

    # Stratified cross validation 
    for train, test in StratifiedKFold(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train], y.iloc[train].values.ravel()) # Fit pipeline to data
        predictions = pipeline.predict(X.iloc[test]) # Make predictions
        truth = y.iloc[test] # Get the true values of y
                    
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions) # Score between true values and predictions of y
            scores.append(score) # Add score to list

    # Log summary of the metrics
    for name, _, scores in metrics:
        mean_score = sum(scores)/number_of_splits # Mean of the scores
        std_score = statistics.stdev(scores) # Variance (standard deviation) of the scores

        # # Log metrics to ML Flow
        mlflow.log_metric(f"mean_{name}", mean_score) # Log mean of metrics 
        mlflow.log_metric(f"std_{name}", std_score) # Log standard deviation of metrics

        # Print scores
        print(f"mean_{name}", mean_score) 