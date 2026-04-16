import os
import requests
import zipfile

def download_tiny_imagenet(url, dest_folder="data"):
    # 1. Creazione cartella destinazione
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    zip_path = os.path.join(dest_folder, "tiny-imagenet-200.zip")
    
    # 2. Download del file (se non esiste già)
    if not os.path.exists(zip_path):
        print("Download in corso (circa 230MB)...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print("File zip già presente, salto il download.")
    
    # 3. Scompattamento
    print("Scompattamento in corso...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    
    # 4. Riordino cartella Validation (Fondamentale per ImageFolder!)
    print("Riordino cartella Validation...")
    val_dir = os.path.join(dest_folder, 'tiny-imagenet-200', 'val')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')

    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) >= 2:
                    fn, cls = parts[0], parts[1]
                    # Crea sottocartella per la classe
                    class_dir = os.path.join(val_dir, cls)
                    os.makedirs(class_dir, exist_ok=True)
                    # Sposta l'immagine
                    src = os.path.join(images_dir, fn)
                    dst = os.path.join(class_dir, fn)
                    if os.path.exists(src):
                        os.rename(src, dst)
        
        # Pulizia: rimuoviamo la cartella images vuota e il file annotazioni
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        print("Riordino completato con successo.")
    else:
        print("Errore: val_annotations.txt non trovato. Forse il dataset è già ordinato?")

    print(f"Dataset pronto e verificato in: {dest_folder}/tiny-imagenet-200")

if __name__ == "__main__":
    DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    download_tiny_imagenet(DATA_URL, dest_folder="data")