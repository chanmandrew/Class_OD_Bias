import numpy as np
import pandas as pd

def split_positives(df, patient_ids):
  # Filter patients by current scope
  patient_df = df[df['patientId'].isin(patient_ids)]
  # Split patients based on ground truth value
  positive_ids = patient_df[patient_df['Target'] == 1]['patientId'].values
  negative_ids = patient_df[patient_df['Target'] == 0]['patientId'].values
  return positive_ids, negative_ids

def random_samples(arr, sizes, seed=1337):
  # Use different seeds for cross-validation
  np.random.seed(seed)
  samples = []
  for size in sizes:
    # Randomly sample elements
    rand_samples = np.random.choice(arr, size, replace=False)
    # Filter out all sampled elements i.e. remove them!
    arr = arr[np.in1d(arr, rand_samples, invert=True)]
    samples += [rand_samples]
  return samples

def create_patient_splits(csv_file, sex_ratio, seed=1337):
  if sex_ratio < 0 or sex_ratio > 1:
    raise ValueError('Invalid sex ratio! Value must be between [0,1]')
  # Load CSV file containing patient id, ground truth, and sex
  df = pd.read_csv(csv_file)
  # Separate male and female patients
  male_ids, female_ids = df[df['Sex'] == 'M']['patientId'], df[df['Sex'] == 'F']['patientId']
  # Determine length of train, validation, and testing set
  # This will result in a roughly 72.74-9.09-18.17% split
  # Weird math since we want 50-50 sex ratio in testing set!
  N_val = int(len(female_ids)*0.1)
  N_test = N_val*2
  N_train = len(female_ids) - N_test
  # print(N_train, N_val, N_test)
  # Split both groups into positive and negatives
  positive_male_ids, negative_male_ids = split_positives(df, male_ids)
  positive_female_ids, negative_female_ids = split_positives(df, female_ids)
  # We need to maintain ratio of positives to negatives across all splits
  disease_ratio = len(positive_female_ids)/len(female_ids)
  # print(disease_ratio)
  # Number of female patients in each split
  N_female_train = int(sex_ratio*N_train)
  N_female_val = int(sex_ratio*N_val)
  N_female_test = int(0.5*N_test)
  # Number of male patients in each split
  N_male_train = N_train - N_female_train
  N_male_val = N_val - N_female_val
  N_male_test = N_test - N_female_test
  # print((N_female_train, N_male_train), (N_female_val, N_male_val), (N_female_test, N_male_test))
  # Split positive male patients using disease ratio
  grouped_positive_male_ids = random_samples(
    positive_male_ids, 
    [
      round(disease_ratio*N_male_test),
      round(disease_ratio*N_male_val),
      round(disease_ratio*N_male_train)
    ],
    seed = seed
  )
  # print([arr.shape[0] for arr in grouped_positive_male_ids])
  # Split negative male patients using disease ratio
  grouped_negative_male_ids = random_samples(
    negative_male_ids, 
    [
      N_male_test - len(grouped_positive_male_ids[0]),
      N_male_val - len(grouped_positive_male_ids[1]),
      N_male_train - len(grouped_positive_male_ids[2]),
    ],
    seed = seed
  )
  # print([arr.shape[0] for arr in grouped_negative_male_ids])
  # Split positive female patients using disease ratio
  grouped_positive_female_ids = random_samples(
    positive_female_ids, 
    [
      round(disease_ratio*N_female_test),
      round(disease_ratio*N_female_val),
      round(disease_ratio*N_female_train)
    ],
    seed = seed
  )
  # print([arr.shape[0] for arr in grouped_positive_female_ids])
  # Split negative female patients using disease ratio
  grouped_negative_female_ids = random_samples(
    negative_female_ids, 
    [
      N_female_test - len(grouped_positive_female_ids[0]),
      N_female_val - len(grouped_positive_female_ids[1]),
      N_female_train -len(grouped_positive_female_ids[2]),
    ], 
    seed = seed
  )
  # print([arr.shape[0] for arr in grouped_negative_female_ids])
  # Let's start building our data splits!
  # Build patient ids for training split and shuffle
  train_ids = np.concatenate((
    grouped_positive_male_ids[2], 
    grouped_negative_male_ids[2],
    grouped_positive_female_ids[2],
    grouped_negative_female_ids[2]
  ))
  np.random.shuffle(train_ids)
  # Build patient ids for validation split and shuffle
  val_ids = np.concatenate((
    grouped_positive_male_ids[1], 
    grouped_negative_male_ids[1],
    grouped_positive_female_ids[1],
    grouped_negative_female_ids[1]
  ))
  np.random.shuffle(val_ids)
  # Build patient ids for testing split and shuffle
  test_ids = np.concatenate((
    grouped_positive_male_ids[0], 
    grouped_negative_male_ids[0],
    grouped_positive_female_ids[0],
    grouped_negative_female_ids[0]
  ))
  np.random.shuffle(test_ids)
  # Viola, done!
  return train_ids, val_ids, test_ids

# Sanity checks below!

# # Get metrics for each split
# df = pd.read_csv('RSNA_pneumonia.csv')
# sex_ratio = 1
# train_ids, val_ids, test_ids = create_patient_splits('RSNA_pneumonia.csv', sex_ratio)
# print(f'Length of dataset splits: ({len(train_ids)}, {len(val_ids)}, {len(test_ids)})')
# train_df, val_df, test_df = df[df['patientId'].isin(train_ids)], df[df['patientId'].isin(val_ids)], df[df['patientId'].isin(test_ids)]
# print(f"Sex Ratio in Train: {len(train_df[train_df['Sex'] == 'F'])/len(train_df)}")
# print(f"Disease Ratio in Train: {len(train_df[train_df['Target'] == 1])/len(train_df)}")
# print(f"Sex Ratio in Val: {len(val_df[val_df['Sex'] == 'F'])/len(val_df)}")
# print(f"Disease Ratio in Val: {len(val_df[val_df['Target'] == 1])/len(val_df)}")
# print(f"Sex Ratio in Test: {len(test_df[test_df['Sex'] == 'F'])/len(test_df)}")
# print(f"Disease Ratio in Test: {len(test_df[test_df['Target'] == 1])/len(test_df)}")

# # Load all sex ratio splits
# splits = []
# for sex_ratio in [0, 0.25, 0.5, 0.75, 1]:
#   splits += [list(create_patient_splits('RSNA_pneumonia.csv', sex_ratio))]
# splits = np.array(splits, dtype=object)