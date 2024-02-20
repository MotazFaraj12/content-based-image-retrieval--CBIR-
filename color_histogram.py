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


def compute_histogram(images, bin):
    database_histograms = []

    for image_name, image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [bin, bin], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        database_histograms.append((image_name, hist))

    return database_histograms


def find_in_both(ground_truth, similar_incidence):
    count = 0
    for image_name_gt, _ in ground_truth:
        for image_name_si, _ in similar_incidence:
            if image_name_gt == image_name_si:
                count += 1
                break
    return count


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


def compute_average(list_of_lists):
    # Convert the list of lists into a NumPy array
    data_array = np.array(list_of_lists)
    # Calculate the mean along axis 0
    average_values = np.mean(data_array, axis=0)
    return average_values


bin_number = 120
thresholds = [0, 5, 8, 10, 15, 20, 25, 30, 35, 50, 55, 100, 150, 200, 250, 300, 350, 500, 550, 600, 650, 800, 850, 900,
              950, 999]

database_images = read_image_file()

start_time = time.time()
database_hists = compute_histogram(database_images, bin_number)
end_time = time.time()

histime = end_time - start_time

ground_truth = []

# Create a list to store true positive rates (sensitivity) and false positive rates (1 - specificity)
avg_tpr_list = []
avg_fpr_list = []

# Create a list to store precision, recall, and F1 score
avg_precision_list = []
avg_recall_list = []
avg_f1_list = []

run_time = []

images_to_show = []
q_to_show = []

for i in range(300, 400):
    img_name = f'{i}.jpg'
    img = cv2.imread(f"dataset\\{i}.jpg")
    ground_truth.append((img_name, img))

for h in range(310, 321):
    start_time = time.time()
    # Load images
    query_image = cv2.imread(f'dataset\\{h}.jpg')

    # Convert to HSV color space
    query_hsv = cv2.cvtColor(query_image, cv2.COLOR_BGR2HSV)
    query_hist = cv2.calcHist([query_hsv], [0, 1], None,
                              [bin_number, bin_number], [0, 180, 0, 256])
    query_hist = cv2.normalize(query_hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compute Euclidean distances
    distances = []

    for image_name, hist in database_hists:
        dis = cv2.norm(query_hist - hist, cv2.NORM_L2)
        distances.append((image_name, dis))

    distances = sorted(distances, key=lambda x: x[1])

    # Create a list to store true positive rates (sensitivity) and false positive rates (1 - specificity)
    tpr_list = []
    fpr_list = []

    # Create a list to store precision, recall, and F1 score
    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in thresholds:

        similar_indices = distances[:threshold]
        not_similar_indices = distances[threshold:]

        if threshold == 5 and h == 310:
            q_to_show = query_image
            images_to_show = distances[:6]

        # Calculate true positive, false positive, and false negative
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

        # Append to the lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        tpr_list.append(trp)
        fpr_list.append(fpr)

    end_time = time.time()

    avg_precision_list.append(precision_list)
    avg_recall_list.append(recall_list)
    avg_f1_list.append(f1_list)
    avg_tpr_list.append(tpr_list)
    avg_fpr_list.append(fpr_list)
    run_time.append((end_time - start_time) + histime)

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
