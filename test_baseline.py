import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import matplotlib.pyplot as plt # AJOUT : Bibliothèque pour l'affichage visuel

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if you want to run on CPU, change 'cuda' -> cpu

test_img_folder = 'LR/baseline/*'
save_folder = 'results/baseline'

# AJOUT : Sécurité pour s'assurer que le dossier de destination existe
os.makedirs(save_folder, exist_ok=True)

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f"\n--- Traitement de l'image {idx} : {base} ---")
    
    # read images
    img_orig = cv2.imread(path, cv2.IMREAD_COLOR) # AJOUT : On garde l'image originale en mémoire pour l'affichage
    img = img_orig * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Reconversion en format image (BGR pour OpenCV)
    output_bgr = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_bgr = (output_bgr * 255.0).round().astype(np.uint8)
    
    # Sauvegarde dans le dossier results
    cv2.imwrite(osp.join(save_folder, '{:s}_rlt.png'.format(base)), output_bgr)

    # ==========================================
    # AJOUT : AFFICHAGE CONSOLE AVANT / APRÈS
    # ==========================================
    # OpenCV lit en BGR, Matplotlib a besoin de RGB pour afficher les bonnes couleurs
    img_lr_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_hr_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_lr_rgb)
    axes[0].set_title(f"AVANT : {base}\n(Basse Résolution)", fontsize=12, color='red')
    axes[0].axis('off')
    
    axes[1].imshow(img_hr_rgb)
    axes[1].set_title(f"APRÈS : Modèle Baseline ESRGAN\n(Super-Résolution x4)", fontsize=12, color='green')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
