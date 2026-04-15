import os
import cv2
import glob

# 1. Définition des dossiers
hr_dir = 'dataset_satellite/HR'
lr_dir = 'dataset_satellite/LR'

# 2. Création du dossier LR s'il n'existe pas
os.makedirs(lr_dir, exist_ok=True)

# 3. Récupération de toutes les images Haute Résolution (.tif)
hr_image_paths = glob.glob(os.path.join(hr_dir, '*.tif'))

print(f"🚀 Début de la conversion : {len(hr_image_paths)} images trouvées.")
print("Cela peut prendre une à deux minutes...")

# 4. Boucle de conversion
compteur = 0
for hr_path in hr_image_paths:
    # Lire l'image HR
    img_hr = cv2.imread(hr_path)

    if img_hr is not None:
        # Récupérer les dimensions (Hauteur, Largeur)
        h, w = img_hr.shape[:2]

        # Diviser par 4 pour la Basse Résolution
        nouvelle_largeur = w // 4
        nouvelle_hauteur = h // 4

        # Réduire l'image avec l'interpolation Bicubique (standard pour ESRGAN)
        img_lr = cv2.resize(img_hr, (nouvelle_largeur, nouvelle_hauteur), interpolation=cv2.INTER_CUBIC)

        # Récupérer le nom du fichier original (ex: airplane00.tif)
        nom_fichier = os.path.basename(hr_path)

        # Sauvegarder dans le dossier LR
        lr_path = os.path.join(lr_dir, nom_fichier)
        cv2.imwrite(lr_path, img_lr)

        compteur += 1

        # Afficher la progression pour ne pas s'inquiéter
        if compteur % 500 == 0:
            print(f"⏳ {compteur} images traitées...")

print(f"✅ Terminé ! {compteur} images ont été converties en Basse Résolution et placées dans {lr_dir}.")
