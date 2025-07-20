import pandas as pd

# The name of your original CSV file
original_file_name = 'search_embeddings\embeddings_machine_learning_attention_is_all_you_need.csv'
# The name of the new, corrected file we will create
corrected_file_name = 'search_embeddings\corrected_for_vectosphere.csv'

try:
    # Read the original CSV file
    df = pd.read_csv(original_file_name)

    print("Successfully read the file. Original columns are:")
    print(df.columns)

    # The first few columns are usually metadata (like text, tokens, etc.)
    # The script assumes the first 3 columns are metadata. Change this if needed.
    # All remaining columns are treated as the embedding dimensions.
    embedding_columns = df.columns[1:]

    # Create a new column named 'embeddings'
    # This joins the values from the embedding columns into a single
    # string that looks like a list, e.g., "[0.1, 0.5, ...]"
    df['embeddings'] = df[embedding_columns].values.tolist()
    df['embeddings'] = df['embeddings'].apply(str)

    # Remove the original, separate embedding columns
    df_corrected = df.drop(columns=embedding_columns)

    # Save the result to a new CSV file
    df_corrected.to_csv(corrected_file_name, index=False)

    print(f"\nSuccessfully created the corrected file: '{corrected_file_name}'")
    print("You can now upload this new file to the visualization website.")

except FileNotFoundError:
    print(f"Error: The file '{original_file_name}' was not found.")
    print("Please make sure the script is in the same folder as your CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")