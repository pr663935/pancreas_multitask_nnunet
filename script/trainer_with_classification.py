
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Union, Tuple, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


# ----------------------------------------------------
# 1. Dual-head model: segmentation + classification
# ----------------------------------------------------
class SegClsUNet(nn.Module):
    def __init__(self, base_unet, n_classes_cls):
        super().__init__()
        self.seg_unet = base_unet
        bottleneck_ch = 320  # Default for ResidualEncoderUNet 3d_fullres
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Linear(bottleneck_ch, n_classes_cls)

        # Directly expose the decoder and encoder to avoid attribute issues
        self.decoder = base_unet.decoder
        self.encoder = base_unet.encoder

    def forward(self, x):
        # Get encoder features for classification
        encoder_features = self.encoder(x)
        bottleneck = encoder_features[-1]

        # Segmentation output
        seg_out = self.seg_unet(x)

        # Classification head
        pooled = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        cls_out = self.cls_head(pooled)

        return seg_out, cls_out


# ----------------------------------------------------
# 2. Simple Custom Trainer - load labels once and add to batches
# ----------------------------------------------------
class TrainerWithClassification(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.classification_loss_weight = 0.2
        self.num_classes_cls = 3

        # Load classification labels once during initialization
        self.class_labels = self._load_classification_labels()

        # Tracking variables
        self.train_cls_predictions = []
        self.train_cls_targets = []
        self.val_cls_predictions = []
        self.val_cls_targets = []
        self.val_whole_dsc_epoch = []
        self.val_lesion_dsc_epoch = []

    def _load_classification_labels(self):
        """Load classification labels from the raw dataset directory"""
        import os
        raw_data_folder = os.environ.get('nnUNet_raw', '/workspace/nnUNet_raw')
        labels_file = Path(raw_data_folder) / "Dataset501_Pancreas" / "classification_labels.json"

        if not labels_file.exists():
            raise FileNotFoundError(f"Classification labels not found at {labels_file}")

        with open(labels_file) as f:
            labels = json.load(f)

        print(f"Loaded {len(labels)} classification labels from {labels_file}")
        return labels

    def build_network_architecture(self, architecture_class_name: str,
                                 arch_kwargs: dict,
                                 arch_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                 num_input_channels: int,
                                 num_output_channels: int,
                                 enable_deep_supervision: bool = True) -> nn.Module:
        # Call parent method to build base network
        network = super().build_network_architecture(
            architecture_class_name, arch_kwargs, arch_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        # Wrap with classification head
        return SegClsUNet(base_unet=network, n_classes_cls=self.num_classes_cls)

    def _add_classification_labels_to_batch(self, batch):
        """Add classification labels to batch based on case IDs"""        
        if 'keys' not in batch:
            print("Warning: No 'keys' found in batch, using default labels")
            batch_size = batch['data'].shape[0] if 'data' in batch else 2
            batch['classification_labels'] = torch.randint(0, 3, (batch_size,), dtype=torch.long)
            return batch

        case_ids = batch['keys']
        if hasattr(case_ids, 'tolist'):
            case_ids = case_ids.tolist()
        elif not isinstance(case_ids, (list, tuple)):
            case_ids = [case_ids]

        cls_labels = []
        for case_id in case_ids:
            case_name = case_id.strip() if isinstance(case_id, str) else str(case_id).strip()

            if case_name in self.class_labels:
                cls_labels.append(self.class_labels[case_name])
            else:
                # Try case-insensitive match
                case_name_lower = case_name.lower()
                matches = [k for k in self.class_labels.keys() if k.lower() == case_name_lower]
                if matches:
                    cls_labels.append(self.class_labels[matches[0]])
                else:
                    print(f"Warning: Case {case_name} not found, using label 0")
                    cls_labels.append(0)

        batch['classification_labels'] = torch.tensor(cls_labels, dtype=torch.long)
        return batch

    def compute_custom_dsc(self, predictions, targets) -> Tuple[float, float]:
        """Compute DSC according to README requirements"""
        # Handle deep supervision - take the first (highest resolution) prediction
        if isinstance(predictions, list):
            predictions = predictions[0]
        if isinstance(targets, list):
            targets = targets[0]

        batch_size = predictions.shape[0]
        whole_pancreas_dsc = []
        lesion_dsc = []

        for i in range(batch_size):
            pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
            target = targets[i].cpu().numpy() if torch.is_tensor(targets[i]) else targets[i]

            # Whole pancreas DSC: np.uint8(label > 0) 
            pred_whole = (pred > 0).astype(np.uint8)
            target_whole = (target > 0).astype(np.uint8)

            intersection_whole = np.sum(pred_whole * target_whole)
            union_whole = np.sum(pred_whole) + np.sum(target_whole)

            if union_whole > 0:
                dsc_whole = 2.0 * intersection_whole / union_whole
            else:
                dsc_whole = 1.0
            whole_pancreas_dsc.append(dsc_whole)

            # Lesion DSC: np.uint8(label==2)
            pred_lesion = (pred == 2).astype(np.uint8)
            target_lesion = (target == 2).astype(np.uint8)

            intersection_lesion = np.sum(pred_lesion * target_lesion)
            union_lesion = np.sum(pred_lesion) + np.sum(target_lesion)

            if union_lesion > 0:
                dsc_lesion = 2.0 * intersection_lesion / union_lesion
            else:
                dsc_lesion = 1.0 if np.sum(target_lesion) == 0 else 0.0
            lesion_dsc.append(dsc_lesion)

        return np.mean(whole_pancreas_dsc), np.mean(lesion_dsc)

    def train_step(self, batch: dict) -> dict:
        # Add classification labels to the batch
        batch = self._add_classification_labels_to_batch(batch)

        data = batch['data']
        target = batch['target']
        cls_target = batch['classification_labels'].to(self.device)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        seg_output, cls_output = self.network(data)
        seg_loss = self.loss(seg_output, target)
        cls_loss = F.cross_entropy(cls_output, cls_target)
        total_loss = seg_loss + self.classification_loss_weight * cls_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        cls_pred = torch.argmax(cls_output, dim=1).cpu().numpy()
        cls_true = cls_target.cpu().numpy()
        self.train_cls_predictions.extend(cls_pred)
        self.train_cls_targets.extend(cls_true)

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        # Add classification labels to the batch
        batch = self._add_classification_labels_to_batch(batch)

        data = batch['data']
        target = batch['target']
        cls_target = batch['classification_labels'].to(self.device)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            seg_output, cls_output = self.network(data)
            seg_loss = self.loss(seg_output, target)
            cls_loss = F.cross_entropy(cls_output, cls_target)
            total_loss = seg_loss + self.classification_loss_weight * cls_loss

            whole_dsc, lesion_dsc = self.compute_custom_dsc(seg_output, target)

            # Compute nnU-Net expected metrics (tp_hard, fp_hard, fn_hard)
            # Use the first output for deep supervision
            if isinstance(seg_output, list):
                output_seg = seg_output[0]
            else:
                output_seg = seg_output

            if isinstance(target, list):
                target_seg = target[0]
            else:
                target_seg = target

            # Get predicted segmentation
            predicted_segmentation_onehot = torch.softmax(output_seg, 1)
            predicted_segmentation = predicted_segmentation_onehot.argmax(1)

            # Compute TP, FP, FN for each class
            axes = tuple(range(1, len(target_seg.shape)))
            tp_hard = torch.zeros((target_seg.shape[0], 3), dtype=torch.float, device=self.device)
            fp_hard = torch.zeros((target_seg.shape[0], 3), dtype=torch.float, device=self.device)
            fn_hard = torch.zeros((target_seg.shape[0], 3), dtype=torch.float, device=self.device)

            for b in range(target_seg.shape[0]):
                for c in range(3):  # num_classes
                    tp_hard[b, c] = torch.sum((predicted_segmentation[b] == c) & (target_seg[b] == c))
                    fp_hard[b, c] = torch.sum((predicted_segmentation[b] == c) & (target_seg[b] != c))
                    fn_hard[b, c] = torch.sum((predicted_segmentation[b] != c) & (target_seg[b] == c))

        cls_pred = torch.argmax(cls_output, dim=1).cpu().numpy()
        cls_true = cls_target.cpu().numpy()
        self.val_cls_predictions.extend(cls_pred)
        self.val_cls_targets.extend(cls_true)

        self.val_whole_dsc_epoch.append(whole_dsc)
        self.val_lesion_dsc_epoch.append(lesion_dsc)

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy(),
        }

    def on_epoch_start(self):
        super().on_epoch_start()
        self.train_cls_predictions = []
        self.train_cls_targets = []
        self.val_cls_predictions = []
        self.val_cls_targets = []
        self.val_whole_dsc_epoch = []
        self.val_lesion_dsc_epoch = []

    def on_epoch_end(self):
        super().on_epoch_end()

        # Classification metrics
        if len(self.train_cls_predictions) > 0:
            train_f1 = f1_score(self.train_cls_targets, self.train_cls_predictions, average='macro', zero_division=0)
            train_acc = accuracy_score(self.train_cls_targets, self.train_cls_predictions)
            print(f"Train Classification - F1: {train_f1:.4f}, Acc: {train_acc:.4f}")

        if len(self.val_cls_predictions) > 0:
            val_f1 = f1_score(self.val_cls_targets, self.val_cls_predictions, average='macro', zero_division=0)
            val_acc = accuracy_score(self.val_cls_targets, self.val_cls_predictions)
            print(f"Val Classification - F1: {val_f1:.4f}, Acc: {val_acc:.4f}")

            # Custom DSC
            if len(self.val_whole_dsc_epoch) > 0:
                avg_whole_dsc = np.mean(self.val_whole_dsc_epoch)
                avg_lesion_dsc = np.mean(self.val_lesion_dsc_epoch)
                print(f"Custom DSC - Whole: {avg_whole_dsc:.4f}, Lesion: {avg_lesion_dsc:.4f}")
                print("Targets: Minreq(Whole:0.85+, Lesion:0.27+, F1:0.6+) | idealreq(Whole:0.91+, Lesion:0.31+, F1:0.7+)")
