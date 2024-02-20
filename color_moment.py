import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import glob

# Motaz Faraj 1190553

def read_image_file():
    folder_path = "dataset"
    image_paths = glob.glob(folder_path + "/*.jpg")
    image_data = []

    for image_path in image_paths:
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is not None:
            image_data.append((img_name, img))
        else:
            print(f"Error reading image: {image_path}")

    return image_data


# Function to compute color moments with specific weights
def compute_color_moments(images, weights):
    color_moments = []

    for image_name, image in images:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        mean_values = np.mean(lab_image, axis=(0, 1))
        std_dev_values = np.std(lab_image, axis=(0, 1))
        skewness_values = np.mean(((lab_image - mean_values) / std_dev_values) ** 3, axis=(0, 1))

        # Normalize the features
        normalized_mean = (mean_values - np.mean(mean_values)) / np.std(mean_values)
        normalized_std_dev = (std_dev_values - np.mean(std_dev_values)) / np.std(std_dev_values)
        normalized_skewness = (skewness_values - np.mean(skewness_values)) / np.std(skewness_values)

        # Apply weights to each moment
        weighted_mean = normalized_mean * weights[0]
        weighted_std_dev = normalized_std_dev * weights[1]
        weighted_skewness = normalized_skewness * weights[2]

        # Concatenate the normalized features into a single vector
        color_moment = np.concatenate((weighted_mean, weighted_std_dev, weighted_skewness))
        color_moments.append((image_name, color_moment))

    return color_moments


# Function to find common elements between two lists of image data
def find_in_both(ground_truth, similar_incidence):
    count = 0
    for image_name_gt, _ in ground_truth:
        for image_name_si, _ in similar_incidence:
            if image_name_gt == image_name_si:
                count += 1
                break
    return count


# Function to find elements in one list that are not present in another list
def find_in_one(array1, array2):
    count = 0
    for image_name_nsi, _ in array1:
        found = False
        for image_name_gt, _ in array2:
            if image_name_nsi == image_name_gt:
                found = True
                break
        if not found:
            count += 1
    return count


# Function to compute the average of a list of lists
def compute_average(list_of_lists):
    # Convert the list of lists into a NumPy array
    data_array = np.array(list_of_lists)
    # Calculate the mean along axis 0
    average_values = np.mean(data_array, axis=0)
    return average_values


# Define thresholds
thresholds = [0, 5, 8, 10, 15, 20, 25, 30, 35, 50, 55, 100, 150, 200, 250, 300, 350, 500, 550, 600, 650, 800, 850, 900,
              950, 999]

# Define weights for different tasks
# to give all moments equal weights use (1.0, 1.0, 1.0)
weights = (9.0, 7.0, 1.0)

# Read image files
database_images = read_image_file()

# Compute color moments
start_time = time.time()
database_hists = compute_color_moments(database_images, weights)
end_time = time.time()

histime = end_time - start_time

# Initialize ground truth
ground_truth = []

# Create lists to store average true positive rates (sensitivity), false positive rates (1 - specificity), precision, recall, and F1 score
avg_tpr_list = []
avg_fpr_list = []
avg_precision_list = []
avg_recall_list = []
avg_f1_list = []

run_time = []

images_to_show = []
q_to_show = []

# Loop through a range of images for query
for i in range(300, 400):
    img_name = f'{i}.jpg'
    img = cv2.imread(f"dataset\\{i}.jpg")
    ground_truth.append((img_name, img))

# Loop through a range of images for query
for h in range(310, 321):
    start_time = time.time()
    # Load query image
    query_image = cv2.imread(f'dataset\\{h}.jpg')

    # Convert query image to LAB color space
    lab_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2LAB)

    # Calculate mean, standard deviation, and skewness of each channel for the query image
    mean_values = np.mean(lab_image, axis=(0, 1))
    std_dev_values = np.std(lab_image, axis=(0, 1))
    skewness_values = np.mean(((lab_image - mean_values) / std_dev_values) ** 3, axis=(0, 1))

    # Normalize the features
    normalized_mean = (mean_values - np.mean(mean_values)) / np.std(mean_values)
    normalized_std_dev = (std_dev_values - np.mean(std_dev_values)) / np.std(std_dev_values)
    normalized_skewness = (skewness_values - np.mean(skewness_values)) / np.std(skewness_values)

    weighted_mean = normalized_mean * weights[0]
    weighted_std_dev = normalized_std_dev * weights[1]
    weighted_skewness = normalized_skewness * weights[2]

    # Concatenate normalized features into a single vector for the query image
    query_color_moment = np.concatenate((weighted_mean, weighted_std_dev, weighted_skewness))

    # Compute Euclidean distances between the query image and database images
    distances = []

    for image_name, moment in database_hists:
        dis = cv2.norm(query_color_moment - moment, cv2.NORM_L2)
        distances.append((image_name, dis))

    distances = sorted(distances, key=lambda x: x[1])

    # Create lists to store true positive rates (sensitivity), false positive rates (1 - specificity), precision,
    # recall, and F1 score for the current query image
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Loop through different thresholds
    for threshold in thresholds:
        similar_indices = distances[:threshold]
        not_similar_indices = distances[threshold:]

        if threshold == 5 and h == 310:
            q_to_show = query_image
            images_to_show = distances[:6]

        # Calculate true positive, false positive, false negative, and true negative
        true_positive = find_in_both(ground_truth, similar_indices)
        false_positive = find_in_one(similar_indices, ground_truth)
        true_negative = find_in_one(not_similar_indices, ground_truth)
        false_negative = find_in_one(ground_truth, similar_indices)

        # Calculate precision, recall, and F1 score
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate sensitivity and specificity
        trp = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0

        # Append values to the lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        tpr_list.append(trp)
        fpr_list.append(fpr)

    end_time = time.time()
    # Append lists for the current query image to the average lists
    avg_precision_list.append(precision_list)
    avg_recall_list.append(recall_list)
    avg_f1_list.append(f1_list)
    avg_tpr_list.append(tpr_list)
    avg_fpr_list.append(fpr_list)
    run_time.append((end_time - start_time) + histime)

# Compute average values for precision, recall, F1 score, true positive rate (sensitivity), and false positive rate (1 - specificity)
precision_list = compute_average(avg_precision_list)
fpr_list = compute_average(avg_fpr_list)
tpr_list = compute_average(avg_tpr_list)
f1_list = compute_average(avg_f1_list)
recall_list = compute_average(avg_recall_list)

# Calculate AUC for precision-recall curve
precision_recall_auc = auc(fpr_list, tpr_list)
print(f'Precision-Recall AUC: {precision_recall_auc:.2f}')
print(f'Max Precision: {max(precision_list)}')
print(f'Average precision: {np.mean(precision_list)}')
print(f'Max Recall: {max(precision_list)}')
print(f'Average recall: {np.mean(recall_list)}')
print(f'Max F1-score: {max(precision_list)}')
print(f'Average F1-score: {np.mean(f1_list)}')
print(f'Time taken for the output: {np.mean(run_time)}')

# Plot Recall curve
plt.figure()
plt.plot(thresholds, recall_list, color='darkorange', lw=2, label='Recall curve')
plt.xlim([0.0, 1000])
plt.ylim([0.0, 1.05])
plt.xlabel('thresholds')
plt.ylabel('Recall')
plt.title('Recall Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision curve
plt.figure()
plt.plot(thresholds, precision_list, color='darkorange', lw=2, label='Precision curve')
plt.xlim([0.0, 1000])
plt.ylim([0.0, 1.05])
plt.xlabel('thresholds')
plt.ylabel('Precision')
plt.title('Precision Curve')
plt.legend(loc="lower right")
plt.show()

# Plot F1 score
plt.figure()
plt.plot(thresholds, f1_list, color='darkorange', lw=2, label='F1 Score')
plt.xlim([0.0, 1000])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2,
         label='ROC curve (area = {:.2f})'.format(auc(fpr_list, tpr_list)))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


cv2.imshow(f"Query image", q_to_show)
cv2.waitKey(0)

for image_name, dis in images_to_show:
    for image_name_database, img in database_images:
        if image_name == image_name_database:
            cv2.imshow(f"Result {image_name}", img)
            cv2.waitKey(0)