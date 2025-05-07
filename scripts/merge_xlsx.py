import os
import pandas as pd

folder_path = "./reports/" 

xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

with pd.ExcelWriter(os.path.join(folder_path, '_Final_Report.xlsx'), engine='openpyxl') as writer:
    for file in xlsx_files:
        sheet_name = os.path.splitext(file)[0][:31]
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Processo de mesclagem concluído.")
