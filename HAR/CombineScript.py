#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   ES335- Machine Learning- Assignment 1
#
# This script combines the data from the UCI HAR Dataset into a more usable format.
# The data is combined into a single csv file for each subject and activity. 
# The data is then stored in the Combined folder.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import pandas as pd
import numpy as np
import os

# Correct paths for Windows
train_path = r'C:\Users\prath\Documents\ML_1\HAR\UCI HAR Dataset\UCI HAR Dataset\train'
test_path  = r'C:\Users\prath\Documents\ML_1\HAR\UCI HAR Dataset\UCI HAR Dataset\test'

# Dictionary of activities. Provided by the dataset.
ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
}

# Helper function to load inertial signals
def load_signals(path, prefix):
    acc_x = pd.read_csv(os.path.join(path, "Inertial Signals", f"total_acc_x_{prefix}.txt"), sep=r'\s+', header=None)
    acc_y = pd.read_csv(os.path.join(path, "Inertial Signals", f"total_acc_y_{prefix}.txt"), sep=r'\s+', header=None)
    acc_z = pd.read_csv(os.path.join(path, "Inertial Signals", f"total_acc_z_{prefix}.txt"), sep=r'\s+', header=None)
    subjects = pd.read_csv(os.path.join(path, f"subject_{prefix}.txt"), sep=r'\s+', header=None)
    labels   = pd.read_csv(os.path.join(path, f"y_{prefix}.txt"), sep=r'\s+', header=None)
    return acc_x, acc_y, acc_z, subjects, labels

# Function to combine data
def combine_data(acc_x, acc_y, acc_z, subjects, labels, dataset_type="Train"):
    for subject in np.unique(subjects.values):
        sub_idxs = np.where(subjects.iloc[:,0] == subject)[0]
        sub_labels = labels.loc[sub_idxs]

        for label in np.unique(sub_labels.values):
            label_idxs = sub_labels[sub_labels.iloc[:,0] == label].index

            # Initialize arrays as None
            accx, accy, accz = None, None, None

            for idx in label_idxs:
                if accx is None:
                    accx = acc_x.loc[idx][64:]
                    accy = acc_y.loc[idx][64:]
                    accz = acc_z.loc[idx][64:]
                else:
                    accx = np.hstack((accx, acc_x.loc[idx][64:]))
                    accy = np.hstack((accy, acc_y.loc[idx][64:]))
                    accz = np.hstack((accz, acc_z.loc[idx][64:]))

            # Make folder if it doesn't exist
            folder_path = os.path.join("Combined", dataset_type, ACTIVITIES[label])
            os.makedirs(folder_path, exist_ok=True)

            # Save CSV
            data = pd.DataFrame({'accx': accx, 'accy': accy, 'accz': accz})
            save_path = os.path.join(folder_path, f"Subject_{subject}.csv")
            data.to_csv(save_path, index=False)

# Combine training data
train_acc_x, train_acc_y, train_acc_z, train_subjects, train_labels = load_signals(train_path, "train")
combine_data(train_acc_x, train_acc_y, train_acc_z, train_subjects, train_labels, "Train")
print("Done combining the training data.")

# Combine testing data
test_acc_x, test_acc_y, test_acc_z, test_subjects, test_labels = load_signals(test_path, "test")
combine_data(test_acc_x, test_acc_y, test_acc_z, test_subjects, test_labels, "Test")
print("Done combining the testing data.")

print("All data combined successfully.")
