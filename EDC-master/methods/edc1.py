# methods/edc1.py
# Updated to log RQASW scale weights (scale_w1/w2/w3) from edc.py every eval.
# All other logic identical to the corrected version.

import contextlib
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, precision_recall_curve,
)
from torch.cuda.amp import autocast, GradScaler
from helper_modules.train_utils import Bn_Controller
import time
from tqdm import tqdm
import cv2

USE_CUDA = torch.cuda.is_available()
USE_MPS  = False


# ---------------------------------------------------------------------------
# Unified timer
# ---------------------------------------------------------------------------
class TimerEvent:
    def __init__(self):
        self.start_time = None
        self.end_time   = None

    def record(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()

    def elapsed_time(self, other):
        return (other.end_time - self.start_time) * 1000   # ms


# ---------------------------------------------------------------------------
# Helper: convert raw xo[i] tensor → (H, W, 3) uint8 numpy for saving
# ---------------------------------------------------------------------------
def to_hwc_uint8(img_np):
    """
    Accepts any of:
      - (C, H, W)  float or uint8   → transpose to (H, W, C)
      - (H, W, C)  float or uint8   → keep as-is
      - (H, W)     grayscale        → expand to (H, W, 3)
    Always returns (H, W, 3) uint8.
    """
    if img_np.ndim == 2:
        # grayscale (H, W) → (H, W, 3)
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.ndim == 3:
        if img_np.shape[0] in (1, 3) and img_np.shape[0] < img_np.shape[1]:
            # CHW → HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[-1] == 1:
            # single-channel HWC → 3-channel
            img_np = np.repeat(img_np, 3, axis=-1)
    # scale float [0,1] to uint8 if needed
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255)
        img_np = img_np.astype(np.uint8)
    return img_np


# ---------------------------------------------------------------------------
# EDC trainer / evaluator
# ---------------------------------------------------------------------------
class EDC:
    def __init__(
        self,
        model,
        it=0,
        num_eval_iter=1000,
        amap_reduction=0.1,
        tb_log=None,
        logger=None,
    ):
        super().__init__()
        self.model         = model
        if USE_MPS:
            self.model = self.model.float()

        self.num_eval_iter = num_eval_iter
        self.tb_log        = tb_log
        self.optimizer     = None
        self.scheduler     = None
        self.it            = 0
        self.logger        = logger
        self.print_fn      = print if logger is None else logger.info
        self.amap_reduction = amap_reduction
        self.bn_controller  = Bn_Controller()

    # ------------------------------------------------------------------
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, args, device, logger=None):
        self.model.train()

        save_path = getattr(args, "seed_save_path", os.path.join(args.save_dir, args.save_name))
        os.makedirs(save_path, exist_ok=True)

        if USE_MPS:
            start_batch = TimerEvent(); end_batch = TimerEvent()
            start_run   = TimerEvent(); end_run   = TimerEvent()
        else:
            start_batch = torch.cuda.Event(enable_timing=True)
            end_batch   = torch.cuda.Event(enable_timing=True)
            start_run   = torch.cuda.Event(enable_timing=True)
            end_run     = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_auc = 0.0
        best_it       = 0
        scaler        = GradScaler()
        train_log     = []

        if args.resume:
            eval_dict = self.evaluate(device=device, args=args, save_visual=True)
            self.print_fn(eval_dict)

        for idx, x, _, y, filename in tqdm(self.loader_dict['train']):

            if self.it > args.num_train_iter:
                break

            if USE_MPS:
                end_batch.end()
            else:
                end_batch.record()
                torch.cuda.synchronize()

            start_run.record()

            x = x.to(device)
            y = y.to(device, dtype=torch.float32)
            if USE_MPS:
                x, y = x.float(), y.float()

            amp_cm = autocast if (args.amp and not USE_MPS) else contextlib.nullcontext

            with amp_cm():
                result     = self.model(x)
                edc_loss   = result['loss'].mean()
                total_loss = edc_loss

            if args.amp and not USE_MPS:
                scaler.scale(total_loss).backward()
                if args.clip > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                edc_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()

            if USE_MPS:
                end_run.end()
            else:
                end_run.record()
                torch.cuda.synchronize()

            tb_dict = {
                'train/total_loss':    total_loss.detach().item(),
                'train/edc_loss':      edc_loss.detach().item(),
                'train/recon_loss':    result['recon_loss'].detach().item(),
                'train/var_loss':      result['var_loss'].detach().item(),
                'train/e1_std':        result['e1_std'].detach().item(),
                'train/e2_std':        result['e2_std'].detach().item(),
                'train/e3_std':        result['e3_std'].detach().item(),
                # RQASW: log current adaptive weights every iteration
                'train/scale_w1':      result['scale_w1'].item(),
                'train/scale_w2':      result['scale_w2'].item(),
                'train/scale_w3':      result['scale_w3'].item(),
                'lr':                  self.optimizer.param_groups[0]['lr'],
                'train/prefetch_time': start_batch.elapsed_time(end_batch) / 1000.,
                'train/run_time':      start_run.elapsed_time(end_run) / 1000.,
            }

            if (self.it + 1) % self.num_eval_iter == 0:
                eval_dict = self.evaluate(device=device, args=args, save_visual=True)

                eval_dict_tb = {
                    k: v for k, v in eval_dict.items()
                    if k not in ('eval/y_true', 'eval/y_score')
                }
                tb_dict.update(eval_dict_tb)

                if tb_dict['eval/AUC'] > best_eval_auc:
                    best_eval_auc = tb_dict['eval/AUC']
                    best_it       = self.it
                    self.save_model('model_best.pth', save_path)
                    self.print_fn(f"  -> New best AUC: {best_eval_auc:.4f} — checkpoint saved.")

                # Log final RQASW weights at each evaluation
                self.print_fn(
                    f"Iter {self.it} | AUC {tb_dict['eval/AUC']:.4f} | "
                    f"RQASW w=["
                    f"{tb_dict['train/scale_w1']:.3f}, "
                    f"{tb_dict['train/scale_w2']:.3f}, "
                    f"{tb_dict['train/scale_w3']:.3f}] | "
                    f"BEST_AUC: {best_eval_auc:.4f} @ iter {best_it}"
                )

                if self.tb_log is not None:
                    self.tb_log.update(tb_dict, self.it)

                tb_dict['it'] = self.it
                train_log.append(tb_dict)

            self.it += 1
            del tb_dict
            start_batch.record()

        log_path = os.path.join(save_path, 'train_log.pkl')
        with open(log_path, 'wb') as f:
            pickle.dump(train_log, f)

        eval_dict = self.evaluate(device=device, args=args)

        self.print_fn("Generating heatmaps (ABNORMAL images only)...")
        self.generate_heatmaps(device, args)

        eval_dict.update({'eval/best_auc': best_eval_auc, 'eval/best_it': best_it})
        return eval_dict

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, device, eval_loader=None, args=None, save_visual=False):
        self.model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        total_num  = 0.0
        total_loss = 0.0
        y_true, y_prob   = [], []
        y1_prob, y2_prob, y3_prob = [], [], []

        for _, x, xo, y, file_names in eval_loader:
            x, y = x.to(device), y.to(device, dtype=torch.float32)
            if USE_MPS:
                x, y = x.float(), y.float()

            num_batch   = x.shape[0]
            total_num  += num_batch
            result      = self.model(x)

            p_img  = self._reduce_map(result['p_all'])
            p1_img = self._reduce_map(result['p1'])
            p2_img = self._reduce_map(result['p2'])
            p3_img = self._reduce_map(result['p3'])

            y_true.extend(y.cpu().tolist())
            y_prob.extend(p_img.cpu().tolist())
            y1_prob.extend(p1_img.cpu().tolist())
            y2_prob.extend(p2_img.cpu().tolist())
            y3_prob.extend(p3_img.cpu().tolist())
            total_loss += result['loss'].detach().item() * num_batch

            if save_visual and args is not None:
                vis_path = os.path.join(args.save_dir, args.save_name, 'heatmap')
                os.makedirs(vis_path, exist_ok=True)

                # Use last 2 spatial dims so it works for both CHW and HWC xo
                h, w = xo.shape[-2], xo.shape[-1]
                anomaly_maps = F.interpolate(
                    result['p_all'], size=(h, w),
                    mode='bilinear', align_corners=False,
                )
                for i in range(xo.shape[0]):
                    # ✅ FIX: convert CHW/grayscale tensor → HWC uint8 numpy
                    image = to_hwc_uint8(xo[i].cpu().numpy())
                    amap  = anomaly_maps[i].cpu().squeeze().numpy()
                    fname = os.path.basename(file_names[i])
                    self.save_anomaly_map(amap, image, vis_path, fname)

        y_true  = np.array(y_true)
        y_prob  = np.array(y_prob)
        y1_prob = np.array(y1_prob)
        y2_prob = np.array(y2_prob)
        y3_prob = np.array(y3_prob)

        def gauss_norm(s):
            return (s - s.mean()) / (s.std() + 1e-8)

        y_prob_n  = gauss_norm(y_prob)
        y1_prob_n = gauss_norm(y1_prob)
        y2_prob_n = gauss_norm(y2_prob)
        y3_prob_n = gauss_norm(y3_prob)

        AUC    = roc_auc_score(y_true, y_prob_n)
        AUC1   = roc_auc_score(y_true, y1_prob_n)
        AUC2   = roc_auc_score(y_true, y2_prob_n)
        AUC3   = roc_auc_score(y_true, y3_prob_n)

        best_auc = max(AUC, AUC1, AUC2, AUC3)
        if best_auc == AUC1:
            y_final = y1_prob_n
        elif best_auc == AUC2:
            y_final = y2_prob_n
        elif best_auc == AUC3:
            y_final = y3_prob_n
        else:
            y_final = y_prob_n

        thresh = return_best_thr(y_true, y_final)
        y_pred = (y_final >= thresh).astype(int)

        acc    = accuracy_score(y_true, y_pred)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        spec   = specificity_score(y_true, y_pred)
        AUC    = roc_auc_score(y_true, y_final)

        self.model.train()
        return {
            'eval/loss':        total_loss / total_num,
            'eval/f1':          f1,
            'eval/recall':      recall,
            'eval/specificity': spec,
            'eval/acc':         acc,
            'eval/AUC':         AUC,
            'eval/AUC1':        AUC1,
            'eval/AUC2':        AUC2,
            'eval/AUC3':        AUC3,
            'eval/best_thr':    thresh,
            'eval/y_true':      y_true,
            'eval/y_score':     y_final,
        }

    # ------------------------------------------------------------------
    # Heatmap generation — ABNORMAL only
    # ------------------------------------------------------------------
    def generate_heatmaps(self, device, args):
        self.model.eval()
        save_path = os.path.join(args.save_dir, args.save_name, 'heatmap')
        os.makedirs(save_path, exist_ok=True)
        n_saved = 0

        with torch.no_grad():
            for _, x, xo, y, file_names in tqdm(
                    self.loader_dict['eval'], desc='Generating heatmaps'):
                x = x.to(device)
                result = self.model(x)

                # Use last 2 spatial dims so it works for both CHW and HWC xo
                h, w = xo.shape[-2], xo.shape[-1]
                anomaly_maps = F.interpolate(
                    result['p_all'], size=(h, w),
                    mode='bilinear', align_corners=False,
                )
                for i in range(xo.shape[0]):
                    label = int(y[i].item()) if torch.is_tensor(y[i]) else int(y[i])
                    if label != 1:
                        continue

                    # ✅ FIX: convert CHW/grayscale tensor → HWC uint8 numpy
                    image = to_hwc_uint8(xo[i].cpu().numpy())
                    amap  = anomaly_maps[i].cpu().squeeze().numpy()
                    fname = os.path.basename(file_names[i])
                    self.save_anomaly_map(amap, image, save_path, fname)
                    n_saved += 1

        self.print_fn(f"Heatmaps saved for {n_saved} ABNORMAL images -> {save_path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, save_name, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'it':        self.it,
        }, os.path.join(save_path, save_name))
        self.print_fn(f"Checkpoint saved: {os.path.join(save_path, save_name)}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        if self.optimizer and checkpoint.get('optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint.get('it', 0)
        self.print_fn(f"Model loaded from {load_path} (iter={self.it})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce_map(self, pmap):
        flat = pmap.flatten(1)
        if isinstance(self.amap_reduction, float):
            k = max(1, int(flat.shape[1] * self.amap_reduction))
            return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        elif self.amap_reduction == 'mean':
            return flat.mean(1)
        else:
            return flat.max(1)[0]

    def save_anomaly_map(self, anomaly_map, image, save_path, file_name):
        """
        anomaly_map : (H, W)      float numpy
        image       : (H, W, 3)   uint8 numpy  — must already be HWC
        """
        # Resize anomaly map to match image if sizes differ
        h, w = image.shape[:2]
        if anomaly_map.shape != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h))

        anomaly_map_norm = min_max_norm(anomaly_map)
        heatmap          = cvt2heatmap(anomaly_map_norm * 255)   # (H,W,3) uint8
        hm_on_img        = heatmap_on_image(heatmap, image)      # (H,W,3) uint8

        base = os.path.splitext(file_name)[0]
        cv2.imwrite(os.path.join(save_path, base + '_overlay.png'), hm_on_img)
        cv2.imwrite(
            os.path.join(save_path, base + '_map.png'),
            (anomaly_map_norm * 255).astype(np.uint8),
        )


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------
def cvt2heatmap(gray):
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


def heatmap_on_image(heatmap, image):
    """
    heatmap : (H, W, 3) uint8
    image   : (H, W, 3) uint8
    Both must be same shape — guaranteed by save_anomaly_map.
    """
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_max == a_min:
        return np.zeros_like(image)
    return (image - a_min) / (a_max - a_min)


def return_best_thr(y_true, y_score):
    from sklearn.metrics import precision_recall_curve
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s  = 2 * precs * recs / (precs + recs + 1e-7)
    f1s  = f1s[:-1]
    mask = ~np.isnan(f1s)
    thrs, f1s = thrs[mask], f1s[mask]
    return float(thrs[np.argmax(f1s)]) if len(f1s) > 0 else 0.5


def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).astype(int)
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    N  = (y_true == 0).sum()
    return float(TN) / (N + 1e-8)


if __name__ == "__main__":
    pass