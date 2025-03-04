import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.nn.functional as F
from tqdm import tqdm


class Metrics:
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='binary'):
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        # Store results
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def plot(self):
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))

        # Plot each metric
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            ax = axs[i//2, i%2]
            values = [res[metric] * 100 for res in self.results.values()]
            ax.bar(self.results.keys(), values)
            ax.set_title(metric)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Percentage')
            # Add values on the bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    
    
# Function to visualise class imbalance over batchers
def visualise_dataloader(dl):
    total_num_images = len(dl.dataset)
    class_0_batch_counts = []
    class_1_batch_counts = []

    graph_df = pd.DataFrame({
        'batch_num':[],
        'class_0':[],
        'class_1':[]
    })

    for i, batch in enumerate(dl):
        
        labels = batch[1].tolist()
        unique_labels = set(labels)
        if len(unique_labels) > 2:
            raise ValueError("More than two classes detected")
        
        class_0_count = labels.count(0)
        class_1_count = labels.count(1)

        class_0_batch_counts.append(class_0_count)
        class_1_batch_counts.append(class_1_count)
        
        graph_df.loc[len(graph_df)] = [i+1, class_0_count, class_1_count]
    
    plt.figure(figsize=(10, 6))

    # Bar width
    bar_width = 0.35

    # Plotting bars for class_1
    plt.bar(graph_df['batch_num'], graph_df['class_1'], bar_width, label='Class 1')

    # Plotting bars for class_0
    plt.bar(graph_df['batch_num'] + bar_width, graph_df['class_0'], bar_width, label='Class 0')

    # Adding labels and title
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Images')
    plt.title('Number of Class 1 and Class 0 for Each Batch Number')
    plt.legend()

    plt.tight_layout()
    plt.show()        

# Convert images back to standard size
def denormalize(image_numpy, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image_numpy = (image_numpy * std + mean)
    return image_numpy

def reverse_transform(image_numpy):
    # Define mean and std used for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Denormalize
    image_numpy = denormalize(image_numpy, mean, std)
    
    # Clip the values to be in the range [0, 1]
    image_numpy = np.clip(image_numpy, 0, 1)
    
    # Convert to PIL image
    #'image_pil = Image.fromarray((image_numpy * 255).astype(np.uint8))
    
    return image_numpy

def evaluate_model_with_images(model, dataloader, threshold=0.5, activation='sigmoid', device='cpu'):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_images = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if activation == 'softmax':
                # Softmax
                probs = F.softmax(outputs, dim=1)  # Get probabilities for each class
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilities of the positive class
                all_labels.extend(labels.cpu().numpy())
                all_images.extend(inputs.cpu().numpy())
            elif activation == 'sigmoid':
                # Sigmodid
                probs = torch.sigmoid(outputs)  # Use sigmoid for binary classification
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_images.extend(inputs.cpu().numpy())
    
    # Reverse transformations for all images
    denormalized_images = [reverse_transform(image.transpose(1, 2, 0)) for image in all_images]
    
    return denormalized_images, all_labels, all_probs


# Find the optimal threshold for F1 score
def find_best_threshold(y_true, probs):
    best_f1 = 0
    best_threshold = 0

    thresholds = np.arange(0, 1.01, 0.01)
    for threshold in thresholds:
        predictions = [1 if prob >= threshold else 0 for prob in probs]
        f1 = f1_score(y_true, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# Define pytorch dataset class
class WarDataset(Dataset):

    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_file)
        self.transform = transform
        
        # # Remove low_confidence_war from data
        # self.img_labels = self.img_labels[self.img_labels['choice'] != 'low_confidence_war']
        self.img_labels = self.img_labels[['image','label_no_lc']]


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        # Convert tensor to PIL image
        image = to_pil_image(image)
        
        # Apply transformations to image
        if self.transform:
            image = self.transform(image)
            
        # Adjust label according to our current schema
        label_mapping = {
            'not_war': 0,
            'military': 1,
            'damaged_infrastructure': 1,
            'military&anchor': 1,
            'damaged_infrastructure&anchor': 1,
            'war':1
        }
        
        if label in label_mapping:
            label = label_mapping[label]

        return image, label

    def __len__(self):
        return len(self.img_labels)
    
    
class EvaluateImages():
    
    def __init__(self, img_arrays, true_labels, pred_labels, probs):
        self.img_arrays = img_arrays
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.probs = probs
        
    # Silent function to add subplot based on image index
    def __add_image(self, i, index, num_images, plotted, probs):
        img = self.img_arrays[i+index]
        
        true_label = 'War related' if self.true_labels[i+index] == 1 else 'Non war related'
        if probs==False:
            predicted_label = 'War related' if self.pred_labels[i+index] == 1 else 'Non war related'
        else:
            predicted_label = round(float(self.probs[i+index]), 3)
        
        # Add subplot
        plt.subplot(2, (num_images + 1) // 2, plotted + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis('off')
        
        
    def plot_images(self, index, num_images=10, probs=False):
        plt.figure(figsize=(20, 10))
        i = -1
        plotted = 0
        while plotted < num_images:
            i+=1
            if probs==False:
                self.__add_image(i, index, num_images, plotted, False)
            else:
                self.__add_image(i, index, num_images, plotted, True)
            plotted +=1

        print(f'Last plotted image index: {i+index}')
        
        
    def plot_false_positives(self, index, num_images=10, probs=False):
        plt.figure(figsize=(20, 10))
        i = -1
        plotted = 0
        while plotted < num_images:
            i+=1
            if (self.true_labels[i+index]==0) and (self.pred_labels[i+index]==1):
                if probs==False:
                    self.__add_image(i, index, num_images, plotted, False)
                else:
                    self.__add_image(i, index, num_images, plotted, True)
                plotted +=1
                
        print(f'Last plotted image index: {i+index}')
        
    def plot_false_negatives(self, index, num_images=10, probs=False):
        plt.figure(figsize=(20, 10))
        i = -1
        plotted = 0
        while plotted < num_images:
            i+=1
            if (self.true_labels[i+index]==1) and (self.pred_labels[i+index]==0):
                if probs==False:
                    self.__add_image(i, index, num_images, plotted, False)
                else:
                    self.__add_image(i, index, num_images, plotted, True)
                plotted +=1
                
        print(f'Last plotted image index: {i+index}')
        
    def plot_true_positives(self, index, num_images=10, probs=False):
        plt.figure(figsize=(20, 10))
        i = -1
        plotted = 0
        while plotted < num_images:
            i+=1
            if (self.true_labels[i+index]==1) and (self.pred_labels[i+index]==1):
                if probs==False:
                    self.__add_image(i, index, num_images, plotted, False)
                else:
                    self.__add_image(i, index, num_images, plotted, True)
                plotted +=1
                
        print(f'Last plotted image index: {i+index}')
        
    def plot_true_negatives(self, index, num_images=10, probs=False):
        plt.figure(figsize=(20, 10))
        i = -1
        plotted = 0
        while plotted < num_images:
            i+=1
            if (self.true_labels[i+index]==0) and (self.pred_labels[i+index]==0):
                if probs==False:
                    self.__add_image(i, index, num_images, plotted, False)
                else:
                    self.__add_image(i, index, num_images, plotted, True)
                plotted +=1
                
        print(f'Last plotted image index: {i+index}')



def occlusion_sensitivity_mask(model, image, patch_size=5):
    """
    Perform occlusion sensitivity analysis with mask sliding over a given image.
    
    Parameters:
    - model: Trained model.
    - image: Preprocessed input image tensor (1, C, H, W).
    - patch_size: Size of the patch to keep.
    
    Returns:
    - heatmap: Occlusion sensitivity heatmap.
    """
    model.eval()  # Set the model to evaluation mode
    _, _, img_height, img_width = image.shape
    heatmap = np.zeros((img_height, img_width))
    
    # Move the image tensor to the correct device
    device = next(model.parameters()).device
    image = image.to(device)
    
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            # Create a masked image
            occluded_image = torch.zeros_like(image)  # Mask initialized with zeros (full occlusion)
            i_end = min(i + patch_size, img_height)
            j_end = min(j + patch_size, img_width)
            occluded_image[:, :, i:i_end, j:j_end] = image[:, :, i:i_end, j:j_end]
            
            # Predict the output for the occluded patch
            with torch.no_grad():
                output = model(occluded_image)
                score = torch.sigmoid(output).item()
            
            # Record the score in the heatmap
            heatmap[i:i_end, j:j_end] = score
            # print(score)
    
    return heatmap
    
def occlusion_sensitivity(model, image, patch_size=50):
    """
    Perform occlusion sensitivity analysis by occluding patches of the input image and measuring the change in model output.
    
    Parameters:
    - model: Trained model.
    - image: Preprocessed input image tensor (1, C, H, W).
    - patch_size: Size of the occlusion patch.
    
    Returns:
    - heatmap: Occlusion sensitivity heatmap.
    """
    model.eval()  # Set the model to evaluation mode
    _, _, img_height, img_width = image.shape
    heatmap = np.zeros((img_height, img_width))
    
    # Move the image tensor to the correct device
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Get the original prediction
    with torch.no_grad():
        original_output = model(image)
        original_score = torch.sigmoid(original_output).item()
    
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            # Create a copy of the image with a masked patch
            occluded_image = image.clone()
            i_end = min(i + patch_size, img_height)
            j_end = min(j + patch_size, img_width)
            occluded_image[:, :, i:i_end, j:j_end] = 0  # Mask with zeros
            
            # Predict the output for the occluded image
            with torch.no_grad():
                output = model(occluded_image)
                occluded_score = torch.sigmoid(output).item()

            # The sensitivity is the change in the score
            sensitivity = original_score - occluded_score
            # Record the sensitivity in the heatmap
            heatmap[i:i_end, j:j_end] = sensitivity
    
    return heatmap

def summa(s,a):
    return(s+a)