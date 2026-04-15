import os
from PIL import Image

source_dir = '/Users/arseniikostin/Downloads/dataset'
target_dir = 'dataset_clean'

classes = ['sad', 'happy', 'angry', 'surprised']
target_size = (224, 224)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print("Начинаю обработку датасета...")

for emotion in classes:
    source_folder = os.path.join(source_dir, emotion)
    target_folder = os.path.join(target_dir, emotion)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    if not os.path.exists(source_folder):
        print(f"Папки {source_folder} не существует, пропускаю.")
        continue

    print(f"Обработка папки: {emotion}...")
    
    for filename in os.listdir(source_folder):
        if filename.startswith('.'): 
            continue
        
        img_path = os.path.join(source_folder, filename)
        
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                new_filename = os.path.splitext(filename)[0] + '.jpg'
                img.save(os.path.join(target_folder, new_filename), 'JPEG', quality=95)
                
        except Exception as e:
            print(f"Ошибка с файлом {filename}: {e}")

print("-" * 30)
print(f"Готово! Чистый датасет лежит тут: {target_dir}")