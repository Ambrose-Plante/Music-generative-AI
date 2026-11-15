# prompt: read the csv file that i just uplaoded

import pandas as pd
df = pd.read_csv('Xenium genes - common genes.csv')
ref = pd.read_csv('XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv')

import pandas as pd

# Assuming df and ref are already defined pandas DataFrames

# Create an empty list to store the results
results = []

# Iterate over each gene name in df
for gene in df.iloc[:, 0]:
    # Check if the gene is in ref
    matching_rows = ref[ref.iloc[:, 0] == gene]
    if not matching_rows.empty:
        # Get the gene name and the 7th column value
        gene_name = matching_rows.iloc[0, 0]
        column_7_value = matching_rows.iloc[0, 6]
        # Append the result to the list
        results.append([gene_name, column_7_value])

# Create a new DataFrame from the results list
result_df = pd.DataFrame(results, columns=['Gene Name', 'Column 7'])

# Display the resulting DataFrame
result_df

result_df.to_csv('overlapping_genes.csv', index=False)

I am
