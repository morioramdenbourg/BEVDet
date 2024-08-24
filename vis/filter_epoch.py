import json
import math
import sys

batch_size = 6
# epoch_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
epoch_values = [0, 1]

def average_values(log_entry, target, should_log=False):
    log_values = []
    for index in range(batch_size):
        if log_entry['mode'] == 'val':
            key = f'task{index}.{target}_val'
        else:
            key = f'task{index}.{target}'
        if should_log:
            log_value = math.log10(log_entry[key])
        else:
            log_value = log_entry[key]
        log_values.append(log_value)
    average_log_value = sum(log_values) / len(log_values)
    return average_log_value

def filter_log_file(input_file_path, output_file_path):
    filtered_entries = []

    try:
        with open(input_file_path, 'r') as file:
            # Read and process each line in the log file
            for line in file:
                try:
                    log_entry = json.loads(line.strip())
                    if 'env_info' in log_entry:
                        filtered_entries.append(log_entry)
                    if log_entry.get('epoch') in epoch_values:
                        average_heatmap = average_values(log_entry, 'loss_heatmap', should_log=True)
                        average_xy_l1 = average_values(log_entry, 'loss_xy', should_log=False)

                        if log_entry['mode'] == 'val':
                            log_entry['loss_heatmap_val'] = average_heatmap
                            log_entry['loss_xy_val'] = average_xy_l1
                        else:
                            log_entry['loss_heatmap'] = average_heatmap
                            log_entry['loss_xy'] = average_xy_l1

                        filtered_entries.append(log_entry)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in line: {line.strip()}", file=sys.stderr)

        # Save the filtered results to a new JSON file
        with open(output_file_path, 'w') as output_file:
            for entry in filtered_entries:
                output_file.write(json.dumps(entry) + '\n')
            print(f"Filtered results saved to {output_file_path}")

    except FileNotFoundError:
        print(f"File not found: {input_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_log.py <input_log_file_path> <output_json_file_path>", file=sys.stderr)
    else:
        input_log_file_path = sys.argv[1]
        output_json_file_path = sys.argv[2]
        filter_log_file(input_log_file_path, output_json_file_path)