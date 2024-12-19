import pandas as pd

def clean_and_process_data(input_path, output_path):
    data = pd.read_csv(input_path,encoding="latin-1")
    data.drop_duplicates(inplace=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    numeric_columns = ['Sales', 'Quantity', 'Profit']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    categorical_columns = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    data.to_csv(output_path,encoding="latin-1", index=False)

input_path = 'Sales_dataset.csv'
output_path = 'SuperStore_Sales_Dataset.csv'
clean_and_process_data(input_path, output_path)
