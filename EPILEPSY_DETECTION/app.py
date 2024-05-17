
 
    
import os
from flask import Flask, render_template, request ,send_from_directory
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import mne
import matplotlib.pyplot as plt
import base64
import io


def get_rawplot_as_base64(raw):
    fig = raw.plot()  # Plot the raw data
    buf = io.BytesIO()
    fig.savefig(buf, format = 'png')  
    buf.seek(0)    
    
    # Convert the bytes buffer to a base64 string
    encoded_plot = base64.b64encode(buf.read()).decode('utf-8')
    
    return encoded_plot

def get_prediction_plot_as_base64(values_to_plot, start_second, end_second):
    #plot the prediction values
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(start_second, end_second + 1), values_to_plot, marker='o', linestyle='-')
    plt.title(f'Values from {start_second} to {end_second} seconds')
    plt.xlabel('Second')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    
    # Convert the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert the bytes buffer to a base64 string
    encoded_plot = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()  # Close the plot to release resources
    
    return encoded_plot


# Your Flask routes go here

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads' 

# Load your model from the .pkl file

model = pickle.load(open("Rclfmodel.pkl", "rb"))

def save_plot_as_image(values_to_plot, plot_file_path, start_second, end_second):
    plt.figure(figsize=(10, 6))
    plt.plot(range(start_second, end_second + 1), values_to_plot, marker='o', linestyle='-')
    plt.title(f'Values from {start_second} to {end_second} seconds')
    plt.xlabel('Second')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    plt.savefig(plot_file_path)  # Save the plot as an image file
    plt.close()  # Close the plot to release resources


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file to a temporary location
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)
            
            # Read the uploaded EDF file from the temporary location
            raw = mne.io.read_raw_edf(upload_path, preload=True)
            
            # Filter the raw data
            raw.filter(l_freq=0.5, h_freq=50)
            
            # Apply notch filter to remove powerline interference
            raw.notch_filter(freqs=60) 
            
            # Get the EEG data
            eeg_data = raw.get_data()
            
            # Set EEG reference
            raw.set_eeg_reference()
            
            # Get the list of available channels
            channels = raw.info["ch_names"]
            
            return render_template('channel.html', channels=channels, file_path=upload_path)
        else:
            return 'No file uploaded'
        
       

#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        selected_channel = request.form['channel']
        start_second = int(request.form['start_second'])
        end_second = int(request.form['end_second'])
        file_path = request.form['file_path']
        
        # Read the uploaded EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Filter the raw data
        raw.filter(l_freq=0.5, h_freq=50)
        
        # Apply notch filter to remove powerline interference
        raw.notch_filter(freqs=60) 
        
        # Get the EEG data for the selected channel and time range
        channel_index = raw.info["ch_names"].index(selected_channel)
        eeg_data = raw.get_data()[channel_index]

        
        #ensure start second and end second as integers
        start_second = int(start_second)
        end_second = int(end_second)
        
        # Extract EEG data for the selected time range
        #start_index = int(start_second * raw.info['sfreq'])
        #end_index = int(end_second * raw.info['sfreq'])
        #user_eeg_data = eeg_data[start_index:end_index]
        
        # Standardize the data (mean=0, std=1)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(eeg_data.reshape(-1, 1))
       
        # Reshape the standardized data into chunks
        chunk_size = 178
        num_chunks = len(standardized_data) // chunk_size
        reshaped_data = standardized_data[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
        
        # Make predictions using your model
        prediction = model.predict(reshaped_data)
        
        a = len(prediction)
       
    
        start_index = start_second - 1
        end_index = end_second
        values_to_plot = prediction[start_index:end_index]
        

        
        # Convert predictions to binary
        binary_predictions = (prediction >= 0.5).astype(int)
        
        # Count the number of seizures and non-seizures
        num_seizures = np.sum(binary_predictions)
        num_non_seizures = len(binary_predictions) - num_seizures
        
        #get the raw plot has a base 64 string
        raw_plot_base64 = get_rawplot_as_base64(raw)
        
        # Get the prediction plot as a base64 string
        prediction_plot_base64 = get_prediction_plot_as_base64(values_to_plot, start_second, end_second)
        
        return render_template('result.html' ,prediction = prediction, channel_index = channel_index, num_seizures=num_seizures, num_non_seizures=num_non_seizures,raw_plot_base64 = raw_plot_base64,prediction_plot_base64=prediction_plot_base64,a = a)
      
    
if __name__ == '__main__':
    app.run(debug=True)
    # Define the upload folder
    
    
    