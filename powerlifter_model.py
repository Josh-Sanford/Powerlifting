import numpy as np
import pandas as pd
import sklearn
import pickle
import sys

class PowerlifterGenderModel:
    
    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict_proba(self, X_new, clean=True, augment=True):
        if clean:
            X_new = self.clean_data(X_new)
            
        if augment:
            X_new = self.engineer_features(X_new)
            
        return X_new, self.model.predict_proba(X_new)
    
    def clean_data(self, df):
        # Drop duplicates
        df.drop_duplicates(inplace=True)
    
        # Drop rows where lifters did not place
        df = df[df.Place != 'DQ']
        df = df[df.Place != 'NS']
        df = df[df.Place != 'DD']
        df = df[df.Place != 'G']

        # Drop rows where all 3 lift attempts for any lift are missing
        df = df[df.Squat1Kg.notna() & df.Squat2Kg.notna() & df.Squat3Kg.notna()]
        df = df[df.Bench1Kg.notna() & df.Bench2Kg.notna() & df.Bench3Kg.notna()]
        df = df[df.Deadlift1Kg.notna() & df.Deadlift2Kg.notna() & df.Deadlift3Kg.notna()]

        # Drop unnecessary columns
        df.drop(['Name', 'AgeClass', 'WeightClassKg', 'Division', 'Squat4Kg', 'Bench4Kg', 
                 'Deadlift4Kg', 'Place', 'Country', 'Federation', 'Date',
                 'MeetCountry', 'MeetState', 'MeetName', 'Event'], axis=1, inplace=True)

        # Convert 'Tested' to indicator variable
        df['Tested'] = (df.Tested == 'Yes').astype(int)

        # Fill missing values in object columns with 'Missing'
        for column in df.select_dtypes(include=['object']):
            df[column].fillna('Missing', inplace=True)

        # Ad-hoc feature engineering for missed lifts
        lifts = ['Squat1Kg', 'Squat2Kg', 'Squat3Kg',
                 'Bench1Kg', 'Bench2Kg', 'Bench3Kg',
                 'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg']

        for lift in lifts:
            df[lift + '_missed'] = (df[lift] < 0).astype(int)

        # Create Total_missed_lifts feature
        df['Total_missed_lifts'] = df['Squat1Kg_missed'] + df['Squat2Kg_missed'] + df['Squat3Kg_missed'] + df['Bench1Kg_missed']                                    + df['Bench2Kg_missed'] + df['Bench3Kg_missed'] + df['Deadlift1Kg_missed']                                                        + df['Deadlift2Kg_missed'] + df['Deadlift3Kg_missed']

        # Flag best lifts when missing
        best_lifts = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']

        for lift in best_lifts:
            df[lift + '_missing'] = df[lift].isnull().astype(int)

        # Change negative numbers to nan
        df.mask(df.select_dtypes(include=['float64']) < 0, other=np.nan, inplace=True)

        # Change nan lifts to 0 to indicate a missed lift, which would result in a 0 score
        lifts = ['Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Best3SquatKg',
                 'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Best3BenchKg',
                 'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Best3DeadliftKg']
        for lift in lifts:
            df[lift].fillna(0, inplace=True)

        # Create indicator variables for missingness in remaining columns
        remaining_columns = ['Age', 'BodyweightKg', 'Wilks', 'McCulloch', 
                         'Glossbrenner', 'IPFPoints']
        for column in remaining_columns:
            df[column + '_missing'] = df[column].isnull().astype(int)

        # Fill missing values in remaining numeric features with 0
        for column in remaining_columns:
            df[column].fillna(0, inplace=True)

        # Return cleaned dataframe
        return df

    def engineer_features(self, df):
        # Feature engineering code from exploratory analysis
        df.Equipment.replace(['Wraps', 'Straps'], 'Raw', inplace=True)
        df.Equipment.replace(['Single-ply', 'Multi-ply'], 'Equipped', inplace=True)

        # Calculate allometric relative strength ratings
        df['Allometric_relative_strength'] = df['TotalKg'] * (df['BodyweightKg']**(-2/3))

        # Convert infinite values to nan
        df.Allometric_relative_strength.replace(np.inf, np.nan, inplace=True)

        # Create feature to flag missingness
        df['Allometric_relative_strength_missing'] = df.Allometric_relative_strength.isnull().astype(int)

        # Now replace nan with 0
        df['Allometric_relative_strength'].fillna(0, inplace=True)

        # Create new dataframe with dummy features (excluding target variable)
        df = pd.get_dummies(df, columns=['Equipment'])

        # Return augmented dataframe
        return df
    
def main(data_location, output_location, model_location, clean=True, augment=True):
    # Read dataset
    df = pd.read_csv(data_location)
    
    # Initialize model
    powerlifter_model = PowerlifterGenderModel(model_location)
    
    # Prediction
    df, pred = powerlifter_model.predict_proba(df)
    pred = [p[1] for p in pred]
    
    # Add prediction to dataset
    df['Prediction'] = pred
    
    # Save new dataset with predictions
    df.to_csv(output_location, index=None)
    
if __name__ == '__main__':
    main( *sys.argv[1:] )