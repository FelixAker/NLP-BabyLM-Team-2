import json

def main():
    # Just change this path to point to your JSON file
    file_path = "/Users/felixaker/NLP-BabyLM-Team-2/BLIMP Results 1M/standard_betterdata_blimp.json" 

    try:
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            scores = json.load(file)

        # Extract values and calculate the average
        values = list(scores.values())
        num_items = len(values)
        
        if num_items == 0:
            print("The file is empty.")
            return

        average = sum(values) / num_items

        # Print the results
        print(f"Loaded file: {file_path}")
        print(f"Total tasks: {num_items}")
        print(f"Average score: {average:.4f} ({average * 100:.2f}%)")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'. Check the path!")
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")

if __name__ == "__main__":
    main()
