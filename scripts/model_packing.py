import os
import shutil
import zipfile

#Procedimento necessário para poder subir o modelo no GitHub, por exceder 100MB.

pretrained_models_dir = './models/sbert_models'
model_to_zip = 'gte-small'

model_dir = os.path.join(pretrained_models_dir, model_to_zip)
zip_file_path = f"{model_dir}.zip"
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, pretrained_models_dir))

# Deleta a pasta original do modelo
shutil.rmtree(model_dir)

print("Processo de compactação concluído.")