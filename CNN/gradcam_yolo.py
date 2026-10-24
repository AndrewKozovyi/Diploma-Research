"""
Grad-CAM для YOLO-класифікатора (yolo11n-cls / yolov8n-cls).

Карта будується по виходу останньої згортки класифікаційної голови (Classify.conv, 7x7 при imgsz=224):
градієнт логіта передбаченого класу -> ваги каналів -> зважена сума активацій.
Червоне = ділянки, що найбільше вплинули на рішення. Корисно, щоб показати,
що модель дивиться на саму кістку, а не на фон чи етикетку з номером.

Запуск окремо:
  python gradcam_yolo.py <best.pt> <папка_з_класами (напр. ...\\clever_split\\test)> <вихідна_папка> [imgsz]
Приклад для обраної моделі:
  python gradcam_yolo.py results_yolo\\yolo_11_selected_9_classes-150-27\\weights\\best.pt ^
         D:\\projects\\Diploma\\datasets\\transfer_learning\\transfer_learning_datasets_filtered_by_5\\test ^
         results_yolo\\yolo_11_selected_9_classes-150-27\\gradcam_test
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms


class YoloGradCAM:
    def __init__(self, weights, imgsz=224):
        self.yolo = YOLO(weights)
        self.names = self.yolo.names
        self.net = self.yolo.model.float().eval()
        self.imgsz = imgsz
        self.tf = classify_transforms(imgsz)
        head = self.net.model[-1]
        self._acts = None
        self._grads = None
        head.conv.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, image_path):
        pil = Image.open(image_path).convert("RGB")
        x = self.tf(pil).unsqueeze(0)
        x.requires_grad_(True)  # гарантує граф обчислень навіть якщо ваги заморожені
        with torch.enable_grad():
            out = self.net(x)
            if isinstance(out, (list, tuple)):  # нові версії: (softmax, logits)
                probs, logits = out[0], out[1]
            else:                               # старі версії: лише softmax
                probs, logits = out, torch.log(out + 1e-12)
            cls = int(probs.argmax(1))
            self.net.zero_grad(set_to_none=True)
            logits[0, cls].backward()

        a = self._acts[0].detach()           # C x h x w
        g = self._grads[0].detach()          # C x h x w
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * a).sum(0)).numpy()
        cam = cam / (cam.max() + 1e-8)

        # зображення рівно в тому вигляді, як його бачить мережа (resize + center crop)
        vis = (x.detach()[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        return vis, cam, cls, float(probs[0, cls])


def overlay(vis_rgb, cam, alpha=0.45):
    h, w = vis_rgb.shape[:2]
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def run_gradcam(weights, data_dir, out_dir, imgsz=224, max_per_class=None):
    os.makedirs(out_dir, exist_ok=True)
    cam_model = YoloGradCAM(weights, imgsz)
    total = correct = 0
    for true_cls in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, true_cls)
        if not os.path.isdir(cdir):
            continue
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for f in files[:max_per_class]:
            vis, cam, pred, conf = cam_model(os.path.join(cdir, f))
            pred_name = cam_model.names[pred]
            ok = pred_name == true_cls
            total += 1
            correct += ok
            side = np.concatenate([cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), overlay(vis, cam)], axis=1)
            side = cv2.resize(side, (side.shape[1] * 2, side.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            label = f"True: {true_cls} | Pred: {pred_name} ({conf:.2f})"
            cv2.rectangle(side, (0, 0), (side.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(side, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (80, 220, 80) if ok else (60, 60, 255), 1, cv2.LINE_AA)
            status = "CORRECT" if ok else "WRONG"
            name = f"{status}_True-{true_cls}_Pred-{pred_name}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), side)
    print(f"Grad-CAM: {total} зображень, правильних {correct} ({correct / max(total, 1):.3f}). Папка: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    run_gradcam(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 224)
