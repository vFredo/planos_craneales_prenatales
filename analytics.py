import matplotlib.pyplot as plt
from ultralytics.utils.plotting import plot_results
import glob
from PIL import Image
import pandas as pd

# Path to the results folder
results_path = "runs/detect/train2"  # Adjust this to your actual path

# Plot results
plot_results(file=results_path+"/results.csv")
plt.show()

# Display images
for img_path in glob.glob(f"{results_path}/*.png"):
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.title(img_path.split('/')[-1])
    plt.axis('off')
    plt.show()

# Display confusion matrix
conf_matrix = Image.open(f"{results_path}/confusion_matrix.png")
plt.figure()
plt.imshow(conf_matrix)
plt.title("Confusion Matrix")
plt.axis('off')
plt.show()

# Display results.csv
results_df = pd.read_csv(f"{results_path}/results.csv")
print(results_df)

# Print best performance
best_performance = results_df.loc[results_df['metrics/mAP50-95(B)'].idxmax()]
print("\nBest performance:")
print(best_performance)
