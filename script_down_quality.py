import cv2
import os
import glob

# 1. Créer le dossier pour vos images originales si vous ne l'avez pas fait
os.makedirs('LR/satellite', exist_ok=True)
os.makedirs('satellite_maps', exist_ok=True)

print("Veuillez importer vos captures d'écran dans le dossier 'satellite_maps/' avant de continuer.")
print("Attente de 15 secondes pour vous laisser le temps d'importer...")
import time
time.sleep(15)

# 2. Chercher toutes les images dans le dossier 'images_originales'
image_paths = glob.glob('satellite_maps/*.*')

if not image_paths:
    print("❌ Aucune image trouvée dans 'satellite_maps/'. Veuillez les importer et relancer ce bloc.")
else:
    print(f"✅ {len(image_paths)} image(s) trouvée(s). Dégradation en cours...")
    
    for img_path in image_paths:
        # Lire l'image
        img = cv2.imread(img_path)
        
        if img is not None:
            # Récupérer le nom du fichier (ex: sat1_original.png)
            filename = os.path.basename(img_path)
            # Créer le nouveau nom (ex: sat1_lr.png)
            new_filename = filename.split('.')[0] + '_lr.png'
            
            # Obtenir les dimensions actuelles
            height, width = img.shape[:2]
            
            # Réduire l'image par 4 (C'est ce qui crée la "Basse Résolution")
            # INTER_CUBIC est une méthode mathématique pour réduire l'image
            img_lr = cv2.resize(img, (width // 4, height // 4), interpolation=cv2.INTER_CUBIC)
            
            # Sauvegarder dans le bon dossier pour le script ESRGAN
            output_path = os.path.join('LR/satellite', new_filename)
            cv2.imwrite(output_path, img_lr)
            print(f"✔️ Image dégradée et sauvegardée : {output_path}")
        else:
            print(f"⚠️ Impossible de lire {img_path}.")
            
    print("🎉 Phase de préparation terminée ! Vos images sont dans LR/satellite/ prêtes pour l'IA.")
