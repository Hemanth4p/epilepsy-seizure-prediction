# Epilepsy Detection Using EEG Scans

Epilepsy, also known as seizure disorder, is a brain condition causing seizures, which may involve loss of awareness and convulsions. SUDEP (Sudden Unexpected Death in Epilepsy) is a significant risk for those with frequent seizures, estimated to affect 1 out of every 1,000 people with epilepsy annually.

## EEG Scans and Epilepsy Detection

Epilepsy is diagnosed using EEG (Electroencephalogram) scans that record brain's electrical activity through electrodes attached to the scalp. Spikes in the EEG over a few seconds indicate a seizure. Reviewing EEG scans manually can be labor-intensive, so machine learning and deep learning algorithms are employed for automatic detection. However, these algorithms can be inefficient if the dataset's sampling rate differs from the training data.

## Pre-Processing and Model Training

We use a dataset of EDF (European Data Format) files with a sampling rate of 178Hz for training a Random Forest algorithm. This algorithm, suitable for classification and regression tasks, processes the EEG data for accurate seizure detection. If a user uploads an EEG file through our user-friendly interface, the model extracts and processes the data to the required 178Hz sampling rate, ensuring efficient prediction.

## User Interface

Our application features a user-friendly interface with the following HTML files:
- `index.html`: Main entry point for users to upload EEG files.
- `channel.html`: Displays channels of EEG data.
- `result.html`: Shows the detection results.

The processed output is presented to the user in an intuitive format for better understanding.

## Quickstart

1. **Upload EEG File:** Use `index.html` to upload your EEG file in EDF format.
2. **View Channels:** Navigate to `channel.html` to see the channels of EEG data.
3. **Get Results:** The processed results will be displayed in `result.html`.

For more details, visit our GitHub repository.

---

This summary provides a concise overview of the project, its purpose, and usage instructions for the GitHub README file.
