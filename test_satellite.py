import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 1. Configuration et création des dossiers ---
input_folder = 'satellite_maps'
lr_folder = 'LR/satellite'
result_folder = 'results/satellite'
model_path = 'models/RRDB_ESRGAN_x4.pth'  # Modèle Baseline ou votre modèle

os.makedirs(lr_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# --- 2. Chargement du modèle ESRGAN ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# --- 3. Traitement en boucle des images ---
image_paths = glob.glob(osp.join(input_folder, '*.*'))

if not image_paths:
    print(f"Aucune image trouvée dans '{input_folder}/'. Veuillez vérifier le dossier.")
else:
    print(f"{len(image_paths)} image(s) en cours de traitement...\n")
    
    print(f"{'Image':<15} | {'PSNR (Avant -> Après)':<22} | {'SSIM (Avant -> Après)':<20}")
    print("-" * 65)
    
    fig_apercu = None

    for idx, img_path in enumerate(image_paths):
        base_name = osp.splitext(osp.basename(img_path))[0]

        img_orig_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_orig_raw is None:
            continue

        # CORRECTION GÉOMÉTRIQUE : FORCER UN MULTIPLE DE 4
        h_raw, w_raw = img_orig_raw.shape[:2]
        height = h_raw - (h_raw % 4)
        width = w_raw - (w_raw % 4)
        
        # On recadre l'image originale pour qu'elle soit parfaite
        img_orig = img_orig_raw[:height, :width]

        # ÉTAPE A : Lecture et Dégradation (Basse Résolution)
        img_lr = cv2.resize(img_orig, (width // 4, height // 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(lr_folder, f"{base_name}_lr.png"), img_lr)

        # ÉTAPE B : Super-Résolution par l'IA (ESRGAN)
        img_lr_tensor = img_lr * 1.0 / 255
        img_lr_tensor = torch.from_numpy(np.transpose(img_lr_tensor[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_lr_tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        output_bgr = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output_bgr = (output_bgr * 255.0).round().astype(np.uint8)
        cv2.imwrite(osp.join(result_folder, f"{base_name}_rlt.png"), output_bgr)

        # ÉTAPE C : Évaluation Mathématique (PSNR & SSIM)
        # On simule un zoom classique (Avant)
        img_lr_zoomed = cv2.resize(img_lr, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Le plantage arrivait ici. Maintenant, img_orig, img_lr_zoomed et output_bgr font EXACTEMENT la même taille !
        psnr_avant = psnr(img_orig, img_lr_zoomed, data_range=255)
        ssim_avant = ssim(img_orig, img_lr_zoomed, data_range=255, channel_axis=2, win_size=11)
        
        psnr_apres = psnr(img_orig, output_bgr, data_range=255)
        ssim_apres = ssim(img_orig, output_bgr, data_range=255, channel_axis=2, win_size=11)

        print(f"{base_name:<15} | {psnr_avant:>5.2f} -> {psnr_apres:>5.2f} dB     | {ssim_avant:>6.4f} -> {ssim_apres:>6.4f}")

        # ÉTAPE D : Mémoriser l'affichage de la PREMIÈRE image
        if idx == 0:
            img_lr_rgb = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
            img_ia_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            fig_apercu, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img_lr_rgb)
            axes[0].set_title(f"1. ENTRÉE DÉGRADÉE\n{img_lr.shape[1]}x{img_lr.shape[0]} px", fontsize=12, color='red')
            axes[0].axis('off')

            axes[1].imshow(img_ia_rgb)
            axes[1].set_title(f"2. RÉSULTAT ESRGAN\n{output_bgr.shape[1]}x{output_bgr.shape[0]} px", fontsize=12, color='green')
            axes[1].axis('off')

            axes[2].imshow(img_orig_rgb)
            axes[2].set_title(f"3. VÉRITÉ ORIGINALE\n{img_orig.shape[1]}x{img_orig.shape[0]} px", fontsize=12, color='blue')
            axes[2].axis('off')
            plt.tight_layout()

    print("-" * 65)
    print("\n Traitement terminé et tableau de scores généré !")
    
    if fig_apercu:
        plt.show()
