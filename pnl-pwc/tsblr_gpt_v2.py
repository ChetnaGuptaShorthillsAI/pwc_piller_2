# import pandas as pd
# excel_file = pd.read_excel("pnl-pwc/amazon_titile_vs_golden_set.xlsx",header= 0)
# # custom_headers = ["a","b","c","d","e","f"]
# print(excel_file)
# import json 
# # first_row = excel_file.to_list()
# # custom_dict = dict(zip(custom_headers,first_row))
# # json_str = json.dumps(first_row)
# # print(json_str)




# ### msin 
# # headers = excel_file.columns.to_list()
# # # headers =  excel_file.iloc[0].to_list()
# # list_of_dicts= excel_file.to_dict('records')
# # first_row = list_of_dicts
# # json_str = json.dumps(first_row)
# # print(json_str)

import pandas as pd
import json

# Read the Excel file
df = pd.read_excel('pnl-pwc/amazon_titile_vs_golden_set.xlsx')

# Convert the DataFrame to a list of dictionaries
data = df.to_dict(orient='records')

# Create a list of dictionaries with the desired structure
result = []
for item in data:
    result.append({key: item[key] for key in item.keys()})

# Specify the output JSON file path
output_file_path = 'output.json'

# Save the JSON data to a file
with open(output_file_path, 'w') as json_file:
    json.dump(result, json_file, indent=2)

# Print a message indicating the successful save
print(f"JSON data has been saved to {output_file_path}")
