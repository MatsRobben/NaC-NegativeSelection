import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# List of test files
base = "english.test"
tests = ["lang/middle-english.txt", "lang/plautdietsch.txt", "tagalog.test", "lang/hiligaynon.txt", "lang/xhosa.txt"]
langs = ["middle-english", "plautdietsch", "tagalog", "hiligaynon", "xhosa"]

# Value for r
r = 3

plt.figure(figsize=(15, 5))

# Iterate over each test file
for idx, test in enumerate(tests):
    X = []
    y = []

    # Command to execute
    command_base = f"java -jar negsel2.jar -self english.train -n 10 -r {r} -c -l < {base}"
    command_test = f"java -jar negsel2.jar -self english.train -n 10 -r {r} -c -l < {test}"

    try:
        # Execute the command and capture both standard output and standard error
        output_base = subprocess.check_output(command_base, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        output_test = subprocess.check_output(command_test, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Extract numbers from the output
        numbers_base = [float(line.strip()) for line in output_base.split("\n") if line.strip()]
        numbers_test = [float(line.strip()) for line in output_test.split("\n") if line.strip()]
        
        # Append numbers to X
        X.extend(numbers_base)
        X.extend(numbers_test)
        
        # Create corresponding labels for y
        y.extend([0] * len(numbers_base))
        y.extend([1] * len(numbers_test))
        
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during command execution
        print("Error executing command:", e.output)

    # Convert X and y to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Sort both X and y based on the values in X
    sort_indices = np.argsort(X)[::-1]
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_sorted, X_sorted)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.subplot(1, len(tests), idx+1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve \n lang = {langs[idx]}, r = {r}')
    plt.legend(loc="lower right")

plt.tight_layout(pad=1)
# plt.subplots_adjust(hspace=10)
plt.savefig(f'plots/roc_lang.png')
plt.close()
