"""
Grad-CAM для YOLO-класифікатора (yolo11n/m-cls, yolov8n-cls).

Методи:
  cam      - карта по виходу Classify.conv (сітка imgsz / 32: 7x7 при 224, 10x10 при 320). Голова
             conv -> GAP -> Linear, тому Grad-CAM тут дорівнює CAM; саме цю карту будує застосунок з ONNX.
  layercam - LayerCAM (relu(градієнт) * активація, поелементно) по шару зі stride 8 (model.4, 40x40 при 320):
             детальніша карта, але потребує градієнтів (лише Python).
  fused    - sqrt(cam x layercam): області, важливі для класу (cam), з деталізацією layercam.
tta=True усереднює карту з картою віддзеркаленого зображення (менше артефактів сітки).

bone_share - частка "маси" карти на кістці (фото мають чорний фон); area - частка кадру, яку займає кістка.
bone_share > area означає, що модель дивиться на кістку більше, ніж рівномірна карта.

Запуск окремо:
  python gradcam_yolo.py <best.pt> <папка_з_класами> <вихідна_папка> [imgsz] [max_per_class]
"""
import os
import sys
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

FINE_LAYER = 4
FOREGROUND_LEVEL = 10


class YoloGradCAM:
    def __init__(self, weights, imgsz=None, tta=False):
        self.yolo = YOLO(weights)
        self.names = self.yolo.names
        self.net = self.yolo.model.float().eval()
        self.imgsz = imgsz or int(self.yolo.model.args.get("imgsz", 224))
        self.tf = classify_transforms(self.imgsz)
        self.tta = tta
        self._acts, self._grads = {}, {}
        self._hook(self.net.model[-1].conv, "cam")
        self._hook(self.net.model[FINE_LAYER], "layercam")

    def _hook(self, module, key):
        def save(m, i, out):
            self._acts[key] = out
            out.register_hook(lambda g: self._grads.__setitem__(key, g))
        module.register_forward_hook(save)

    def _maps(self, x, cls=None):
        x = x.clone().requires_grad_(True)  # граф обчислень навіть із замороженими вагами
        with torch.enable_grad():
            out = self.net(x)
            probs, logits = out if isinstance(out, (list, tuple)) else (out, torch.log(out + 1e-12))
            cls = int(probs.argmax(1)) if cls is None else cls
            self.net.zero_grad(set_to_none=True)
            logits[0, cls].backward()
        a, g = self._acts["cam"][0].detach(), self._grads["cam"][0].detach()
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0))
        a, g = self._acts["layercam"][0].detach(), self._grads["layercam"][0].detach()
        layercam = (torch.relu(g) * a).sum(0).clamp(min=0)
        return {"cam": cam, "layercam": layercam}, cls, float(probs[0, cls].detach())

    def _input(self, image_path):
        x = self.tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
        vis = (x[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)  # як бачить мережа
        return x, vis

    def maps(self, image_path, cls=None):
        x, vis = self._input(image_path)
        maps, cls, conf = self._maps(x, cls)
        if self.tta:
            flipped, _, _ = self._maps(torch.flip(x, dims=[3]), cls)
            maps = {k: _norm(v) + _norm(torch.flip(flipped[k], dims=[1])) for k, v in maps.items()}
        size = (self.imgsz, self.imgsz)
        out = {k: _norm(F.interpolate(v[None, None], size=size, mode="bilinear", align_corners=False)[0, 0])
               for k, v in maps.items()}
        out["fused"] = _norm(torch.sqrt(out["cam"] * out["layercam"]))
        return vis, {k: v.numpy() for k, v in out.items()}, cls, conf

    def __call__(self, image_path):
        x, vis = self._input(image_path)
        maps, cls, conf = self._maps(x)
        return vis, _norm(maps["cam"]).numpy(), cls, conf


def _norm(t):
    return t / (t.max() + 1e-8)


def bone_share(vis, cam):
    mask = vis.max(2) > FOREGROUND_LEVEL
    return float((cam * mask).sum() / (cam.sum() + 1e-8)), float(mask.mean())


def overlay(vis_rgb, cam, alpha=0.45):
    h, w = vis_rgb.shape[:2]
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def _caption(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def run_gradcam(weights, data_dir, out_dir, imgsz=None, max_per_class=None, tta=True):
    os.makedirs(out_dir, exist_ok=True)
    cam_model = YoloGradCAM(weights, imgsz, tta=tta)
    rows = []
    for true_cls in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, true_cls)
        if not os.path.isdir(cdir):
            continue
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_aug_" not in f)
        for f in files[:max_per_class]:
            vis, maps, pred, conf = cam_model.maps(os.path.join(cdir, f))
            pred_name = cam_model.names[pred]
            ok = pred_name == true_cls
            row = {"class": true_cls, "file": f, "pred": pred_name, "conf": round(conf, 4), "correct": ok}
            for k, m in maps.items():
                row[f"{k}_bone_share"], row["area"] = bone_share(vis, m)
            rows.append(row)
            panels = [_caption(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), f"True: {true_cls}"),
                      _caption(overlay(vis, maps["cam"]), f"CAM (app) | {pred_name} {conf:.2f}",
                               (80, 220, 80) if ok else (60, 60, 255)),
                      _caption(overlay(vis, maps["fused"]), "sqrt(CAM x LayerCAM)")]
            side = np.concatenate(panels, axis=1)
            side = cv2.resize(side, (side.shape[1] * 2, side.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            name = f"{'CORRECT' if ok else 'WRONG'}_True-{true_cls}_Pred-{pred_name}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), side)
    with open(os.path.join(out_dir, "cam_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"Grad-CAM: {len(rows)} зображень, правильних {correct} ({correct / max(len(rows), 1):.3f}). Папка: {out_dir}")
    for k in ("cam", "layercam", "fused"):
        share = np.mean([r[f"{k}_bone_share"] for r in rows])
        print(f"  {k:9} bone_share {share:.3f} (area {np.mean([r['area'] for r in rows]):.3f})")
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    run_gradcam(sys.argv[1], sys.argv[2], sys.argv[3],
                int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None,
                int(sys.argv[5]) if len(sys.argv) > 5 else None)
