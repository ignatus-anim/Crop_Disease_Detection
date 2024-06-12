import os
import json
import matplotlib.pyplot as plt

# Directory containing the history JSON files
history_dir = 'history'

# Iterate over all .json files in the history directory
for filename in os.listdir(history_dir):
    if filename.endswith('.json'):
        crop_name = filename.split('.')[0]
        file_path = os.path.join(history_dir, filename)
        
        # Load the history data from the JSON file
        with open(file_path, 'r') as file:
            history = json.load(file)
        
        # Extract data from history
        accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(accuracy) + 1)
        
        # Plot Accuracy
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
        plt.title(f'Training and Validation Accuracy - {crop_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
        plt.title(f'Training and Validation Loss - {crop_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save the plots
        output_filename = f'{crop_name}.png'
        plt.savefig(output_filename)
        plt.close()
