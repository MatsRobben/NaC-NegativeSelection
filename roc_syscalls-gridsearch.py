import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool


class Paths:
    train_path = "syscalls/snd-cert/snd-cert.train"
    test_paths = ["syscalls/snd-cert/snd-cert.1.test","syscalls/snd-cert/snd-cert.2.test","syscalls/snd-cert/snd-cert.3.test"]
    label_paths = ["syscalls/snd-cert/snd-cert.1.labels","syscalls/snd-cert/snd-cert.2.labels","syscalls/snd-cert/snd-cert.3.labels"]

    alphabed_path = "syscalls/snd-cert/snd-cert.alpha"
    train_file = "syscalls/snd-cert/temp/snd-cert-chunked.train"

def split_line_into_chunks(line, chunk_length, overlap=False, skip=1):
    chunks = []
    start = 0
    while start < (len(line) - chunk_length + 1):
        chunks.append(line[start:start + chunk_length])
        if overlap:
            start += skip 
        else:
            start += chunk_length
    return chunks

def create_train_chucks(n, r,overlap=True):
    try:
        with open(Paths.train_path, 'r') as file:
            file_content = file.read()

            text = ''
            shortest_train_line_length = float('inf')

            for line in file_content.split('\n'):
                chunks = split_line_into_chunks(line, n, overlap=overlap)

                text += '\n'.join(chunks) + '\n'

                line_length = len(line)
                if line_length < shortest_train_line_length and not line_length == 0:
                    shortest_train_line_length = line_length
            
            # print("Shortest line length in train file:", shortest_train_line_length)

            text = text.rstrip('\n')

            chunked_train_file = f"syscalls/snd-cert/temp/snd-cert-chunked-r:{r}-n:{n}.train"

            with open(chunked_train_file, 'w') as output_file:
                output_file.write(text)
    
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

def load_test(test_path, max=None):
    test_text = []

    try:
        with open(test_path, 'r') as file:
            file_content = file.read().strip()
            test_text = file_content.split('\n')

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

    if max:
        return test_text[:max]
    return test_text

def load_labels(label_path, max=None):
    labels = []

    try:
        with open(label_path, 'r') as label_file:
            for line in label_file:
                label = line.strip()  # Remove leading/trailing whitespaces
                labels.append(int(label))
    except FileNotFoundError:
        print("Label file not found.")
    except Exception as e:
        print("An error occurred while reading labels:", e)

    if max:
        return np.array(labels[:max])
    return np.array(labels)

def get_scores(n, r, test_text):
    train_file = f"syscalls/snd-cert/temp/snd-cert-chunked-r:{r}-n:{n}.train"

    command = f"java -jar negsel2.jar -alphabet file://{Paths.alphabed_path} -self {train_file} -n {n} -r {r} -c -l"
    p = subprocess.Popen(command.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    results = []

    for idx, line in enumerate(test_text):
        print(f'{idx}/{len(test_text)}')
        p.stdin.write(line + '\n')
        p.stdin.flush()
        output = p.stdout.readline().strip().split(' ')
        try:
            output = [float(item) for item in output]
        except:
            print(output)
        results.append(np.mean(output))


    p.stdin.close()
    return_code = p.wait()

    error_output = p.stderr.read()
    if error_output:
        print("Error output:", error_output)

    return np.array(results)

def grid_search_single(params):
    r, n, test_text, labels = params
    
    if r > n:
        return (r, n, 0.0)
    
    create_train_chucks(n=n, r=r, overlap=True)
    
    # Execute the test instances and extract scores
    scores = get_scores(n=n, r=r, test_text=test_text)
            
    # Calculate AUC score
    X = scores
    y = labels
 
    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)

    print(f'n: {n}, r: {r}, AUC: {roc_auc}')
    
    return (r, n, roc_auc)

def grid_search_parallel(r_values, n_values, test_text, labels):
    params_list = [(r, n, test_text, labels) for r in r_values for n in n_values]
    
    with Pool() as pool:
        results = pool.map(grid_search_single, params_list)

    auc_scores = np.zeros((len(r_values), len(n_values)))
    for r, n, auc_score in results:
        auc_scores[r_values.index(r), n_values.index(n)] = auc_score

    return results, auc_scores

def plot_grid(auc_scores):
    # Plot the results as a heatmap
    plt.figure(figsize=(10, 6))
    heatmap = plt.imshow(auc_scores, cmap='YlGn', interpolation='nearest')

    # Add color bar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('AUC Score', fontsize=14)

    # Add annotations with different colors based on AUC score
    for i in range(len(r_values)):
        for j in range(len(n_values)):
            auc_score = auc_scores[i, j]
            text_color = 'white' if auc_score >= 0.5 else 'black'  # Different color based on AUC score
            plt.text(j, i, f'{auc_score:.2f}', ha='center', va='center', color=text_color, fontsize=12)  # Adjust fontsize here

    # Add grid overlay
    plt.grid(visible=True, linestyle='--', alpha=0.5)

    # Customize ticks and labels
    plt.xticks(np.arange(len(n_values)), n_values, fontsize=12)  # Adjust fontsize here
    plt.yticks(np.arange(len(r_values)), r_values, fontsize=12)  # Adjust fontsize here
    plt.xlabel('n values', fontsize=14)  # Adjust fontsize here
    plt.ylabel('r values', fontsize=14)  # Adjust fontsize here
    plt.title('AUC Scores, overlap \n syscalls snd-cert', fontsize=18)  # Adjust fontsize here

    # Move x-axis ticks to the top
    plt.tick_params(axis='x', which='both', bottom=False, top=True)

    # Save the plot and close it
    plt.savefig(f'plots/roc_syscalls_gid_cert_overlap.png', bbox_inches='tight')
    plt.close()

def grid_search(r_values, n_values):
    test_text = load_test(Paths.test_paths[0])

    labels = load_labels(Paths.label_paths[0])

    results, auc_scores = grid_search_parallel(r_values, n_values, test_text, labels)

    plot_grid(auc_scores)

def process_test_instance(test_label_tuple):
    test_path, label_path = test_label_tuple
    text = load_test(test_path=test_path, max=100)
    X = get_scores(n, r, text)
    y = load_labels(label_path, max=100)
    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc    

def test_instances(r,n):
    create_train_chucks(n, r, overlap=True)

    test_label_tuples = zip(Paths.test_paths, Paths.label_paths)

    # Create a multiprocessing pool
    pool = Pool()

    # Process test instances in parallel
    results = pool.map(process_test_instance, test_label_tuples)

    # Plot ROC curves for each test instance
    plt.figure(figsize=(15, 5))
    for idx, (fpr, tpr, roc_auc) in enumerate(results):
        plt.subplot(1, len(Paths.test_paths), idx+1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve, with overlap \n umn, test={idx+1}, r={r}, n={n}')
        plt.legend(loc="lower right")

    plt.tight_layout(pad=1)
    plt.savefig(f'plots/roc_syscalls_cert_overlap.png')
    plt.close()

# Example usage:
r_values = [1, 2, 3, 4, 5, 6, 7]  # Example values for r
n_values = [1, 2, 3, 4, 5, 6, 7]  # Example values for n

# grid_search(r_values, n_values)

r, n = 7, 7

test_instances(r,n)
