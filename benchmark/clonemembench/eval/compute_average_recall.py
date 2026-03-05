import json
import os


root = 'checkpoints/new_data/flatten_llama_openai/recall_metrics'
pattern = ''


all_metrics = []
for filename in os.listdir(root):
    if not filename.endswith('.json'):
        continue
    if pattern and pattern not in filename:
        continue
    filepath = os.path.join(root, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
        all_metrics.append(metrics)

        print(f"File: {filename}")
        for k, k_metrics in metrics.items():
            print(f"@K={k[1:]}:")
            for metric_name, value in k_metrics.items():
                print(f"  {metric_name}: {value:.4f}")

# Aggregate metrics
aggregate = {}
for metrics in all_metrics:
    for k, k_metrics in metrics.items():
        if k not in aggregate:
            aggregate[k] = {}
        for metric_name, value in k_metrics.items():
            if metric_name not in aggregate[k]:
                aggregate[k][metric_name] = []
            aggregate[k][metric_name].append(value)
# Compute averages
average_metrics = {}
for k, k_metrics in aggregate.items():
    average_metrics[k] = {}
    for metric_name, values in k_metrics.items():
        average = sum(values) / len(values) if values else 0.0
        average_metrics[k][metric_name] = average
# Print average metrics
print("Average Recall Metrics Across All Users:")
for k, k_metrics in average_metrics.items():
    print(f"\n@K={k[1:]}:")
    for metric_name, avg_value in k_metrics.items():
        print(f"  {metric_name}: {avg_value:.4f}")