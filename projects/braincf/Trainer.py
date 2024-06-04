from fileinput import filename
import logging
import os
from time import time

from torch.cuda.amp import GradScaler, autocast

import wandb
from core.Trainer import Trainer
from net_utils.simplex_noise import generate_noise
from optim.losses.image_losses import *
from optim.losses.ln_losses import *
import cv2
import copy
from dl_utils.mask_utils import binarize_mask
import matplotlib.pyplot as plt
from dl_utils.radnet_utils import compute_fid, compute_msssim, compute_ssim

from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, join
import torch
from optim.metrics import compute_dice, dice_coefficient_batch, dice_coefficient_batch_
import numpy as np
from dl_utils.fid_score import save_fid_stats, calculate_fid_given_images

PROPS = {
    "sitk_stuff": {
        "spacing": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 1.0),
    },
    "spacing": [999.0, 1.0, 1.0],
}


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params["val_interval"]

        # fixed bootstrapping indices to compute FID
        self.bootstrap_indices = [[39, 23, 32, 36, 39, 52, 46, 23, 50, 7, 40, 27, 26, 36, 24, 20, 31, 0, 34, 28, 25, 33, 26, 11, 43, 42, 44, 45, 24, 21, 3, 14, 33, 52, 36, 15, 52, 40, 22, 32, 51, 29, 37, 16, 27, 50, 19, 40, 52, 25, 29, 49, 6], [38, 5, 5, 5, 24, 7, 32, 14, 44, 41, 27, 41, 52, 26, 25, 50, 52, 1, 10, 18, 26, 50, 52, 14, 30, 41, 20, 36, 28, 6, 8, 44, 27, 24, 3, 3, 22, 39, 11, 14, 18, 52, 16, 36, 45, 45, 50, 38, 10, 47, 41, 6, 13], [15, 15, 22, 7, 52, 15, 10, 13, 32, 15, 3, 46, 15, 51, 28, 32, 38, 11, 0, 7, 32, 15, 20, 33, 16, 34, 7, 17, 10, 40, 19, 11, 41, 41, 16, 20, 17, 11, 47, 18, 38, 23, 22, 1, 2, 38, 18, 48, 40, 15, 29, 35, 18], [48, 24, 52, 11, 36, 27, 48, 34, 52, 19, 21, 28, 40, 12, 47, 27, 23, 30, 49, 40, 32, 32, 18, 15, 0, 49, 36, 14, 49, 19, 2, 26, 41, 6, 1, 26, 4, 37, 39, 26, 15, 36, 6, 49, 19, 0, 35, 36, 12, 33, 29, 2, 44], [31, 24, 23, 52, 32, 52, 17, 47, 52, 21, 25, 43, 39, 45, 13, 15, 45, 38, 10, 13, 50, 40, 52, 50, 32, 26, 17, 5, 9, 51, 22, 26, 50, 33, 38, 15, 48, 25, 8, 43, 48, 0, 26, 3, 34, 38, 34, 51, 48, 21, 36, 52, 10], [14, 43, 43, 45, 25, 35, 17, 26, 13, 42, 48, 29, 37, 12, 24, 52, 47, 50, 47, 44, 14, 29, 2, 36, 33, 4, 46, 3, 48, 26, 40, 3, 22, 26, 40, 6, 7, 17, 0, 35, 52, 22, 20, 10, 47, 1, 44, 11, 30, 11, 19, 12, 29], [44, 38, 30, 4, 25, 28, 15, 42, 22, 11, 33, 27, 25, 25, 23, 1, 19, 18, 0, 6, 42, 33, 2, 52, 41, 16, 29, 46, 9, 0, 31, 32, 43, 13, 17, 0, 25, 13, 10, 22, 36, 14, 25, 20, 29, 28, 36, 43, 28, 2, 12, 52, 50], [0, 21, 35, 14, 7, 10, 52, 13, 19, 40, 35, 44, 24, 32, 21, 20, 30, 23, 46, 21, 30, 19, 16, 12, 10, 7, 4, 8, 30, 37, 5, 51, 20, 51, 8, 11, 23, 40, 14, 20, 0, 24, 27, 7, 24, 28, 52, 30, 18, 16, 2, 24, 10], [38, 49, 38, 12, 8, 16, 10, 38, 14, 51, 43, 25, 29, 37, 4, 49, 10, 12, 25, 18, 10, 3, 9, 11, 2, 21, 32, 48, 22, 37, 50, 25, 27, 45, 2, 43, 44, 31, 52, 23, 39, 24, 49, 27, 7, 0, 6, 18, 15, 15, 23, 8, 21], [27, 46, 28, 13, 20, 41, 17, 6, 48, 8, 32, 13, 0, 46, 0, 35, 50, 23, 42, 0, 0, 28, 36, 12, 51, 34, 52, 43, 36, 25, 40, 11, 48, 14, 47, 37, 29, 12, 22, 45, 29, 20, 47, 46, 30, 34, 8, 45, 4, 49, 7, 36, 33]]
        self.bootstrap_indices = torch.tensor(self.bootstrap_indices)

        self.radnet = torch.hub.load(
            "Warvito/radimagenet-models:main",
            model="radimagenet_resnet50",
            verbose=True,
        )
        self.radnet.to(device)
        self.radnet.eval()

        self.nnunet = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda", 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        nnUNet_results = "/home/malek/mock/autoDDPM/nnunet_data_new/nnunet_results"

        self.nnunet.initialize_from_trained_model_folder(
            join(nnUNet_results, "Dataset500_ATLAS/nnUNetTrainer__nnUNetPlans__2d"),
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name="checkpoint_final.pth",
        )

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer

        epoch_losses = []

        self.early_stop = False
        # to handle loss with mixed precision training
        scaler = GradScaler()

        for epoch in range(self.training_params["nr_epochs"]):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info(
                    "[Trainer::test]: ################ Finished training (early stopping) ################"
                )
                break
            start_time = time()
            batch_loss, count_images = 1.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)

                count_images += images.shape[0]
                transformed_images = (
                    self.transform(images) if self.transform is not None else images
                )

                self.optimizer.zero_grad()

                # for mixed precision training
                with autocast(enabled=True):
                    # Create timesteps
                    timesteps = torch.randint(
                        0,
                        self.model.train_scheduler.num_train_timesteps,
                        (transformed_images.shape[0],),
                        device=images.device,
                    ).long()

                    # Generate random noise and noisy images
                    noise = generate_noise(
                        self.model.train_scheduler.noise_type,
                        images,
                        self.model.train_scheduler.num_train_timesteps,
                    )

                    if self.training_params["training_mode"] == "semantic synthesis":
                        # Get model prediction
                        pred = self.model(
                            inputs=transformed_images,
                            patho_masks=patho_masks,
                            brain_masks=brain_masks,
                            noise=noise,
                            timesteps=timesteps,
                        )
                    elif self.training_params["training_mode"] == "palette training":
                        palette_masks = data[4].to(self.device)
                        pred = self.model(
                            inputs=transformed_images,
                            patho_masks=palette_masks,
                            noise=noise,
                            timesteps=timesteps,
                        )
                    else:
                        # Get model prediction
                        pred = self.model(
                            inputs=transformed_images, noise=noise, timesteps=timesteps
                        )

                    target = (
                        transformed_images
                        if self.model.prediction_type == "sample"
                        else noise
                    )

                    # print("prediction size", pred.size())
                    # print("target size", target.size())
                    loss = self.criterion_rec(pred.float(), target.float())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                batch_loss += loss.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print(
                "Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples".format(
                    epoch, epoch_loss, end_time - start_time, count_images
                )
            )
            wandb.log({"Train/Loss_": epoch_loss, "_step_": epoch})

            # Save latest model
            torch.save(
                {
                    "model_weights": self.model.state_dict(),
                    "optimizer_weights": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(self.client_path, "latest_model.pt"),
            )
            if epoch == 1000 or epoch == 1500 or epoch == 1800:
                torch.save(
                    {
                        "model_weights": self.model.state_dict(),
                        "optimizer_weights": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.client_path, f"model_{epoch}.pt"),
                )
            # Run validation
            if (epoch + 1) % self.val_interval == 0 and epoch > 0:
                self.test(
                    self.model.state_dict(),
                    self.val_ds,
                    "Val",
                    self.optimizer.state_dict(),
                    epoch,
                )

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task="Val", opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """

        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + "_loss_rec": 0,
            task + "_loss_mse": 0,
            task + "_loss_pl": 0,
        }
        test_total = 0

        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)

                b, _, _, _ = x.shape
                test_total += b

                if self.training_params["training_mode"] == "semantic synthesis":
                    x_, _ = self.test_model.sample_from_image(
                        x,
                        patho_masks,
                        brain_masks,
                        noise_level=self.model.noise_level_recon,
                    )
                elif self.training_params["training_mode"] == "palette training":
                    palette_masks = data[4].to(self.device)
                    x_ = self.test_model.palette(x, palette_masks)

                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + "_loss_rec"] += loss_rec.item() * x.size(0)
                metrics[task + "_loss_mse"] += loss_mse.item() * x.size(0)
                metrics[task + "_loss_pl"] += loss_pl.item() * x.size(0)

                for batch_idx in range(b):
                    rec = x_[batch_idx].detach().cpu().numpy()
                    rec[0, 0], rec[0, 1] = 0, 1
                    img = x[batch_idx].detach().cpu().numpy()
                    img[0, 0], img[0, 1] = 0, 1

                    brain_mask = brain_masks[batch_idx].detach().cpu().numpy()
                    patho_mask = patho_masks[batch_idx].detach().cpu().numpy()

                    if self.training_params["training_mode"] == "palette training":
                        print("shape of img", img.shape)
                        print("shape of palette_mask", palette_masks[batch_idx].shape)
                        print("shape of brain_mask", brain_mask.shape)
                        print("shape of rec", rec.shape)
                        palette_mask = palette_masks[batch_idx].detach().cpu().numpy()
                        grid_image = np.hstack([img, palette_mask, rec])
                    else:
                        grid_image = np.hstack([img, patho_mask, brain_mask, rec])

                    wandb.log(
                        {
                            task
                            + "/Example_": [
                                wandb.Image(
                                    grid_image, caption="Iteration_" + str(epoch)
                                )
                            ]
                        }
                    )

        for metric_key in metrics.keys():
            metric_name = task + "/" + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, "_step_": epoch})
        wandb.log({"lr": self.optimizer.param_groups[0]["lr"], "_step_": epoch})
        epoch_val_loss = metrics[task + "_loss_rec"] / test_total
        if task == "Val":
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save(
                    {
                        "model_weights": model_weights,
                        "optimizer_weights": opt_weights,
                        "epoch": epoch,
                    },
                    os.path.join(self.client_path, "best_model.pt"),
                )
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)

    def sdedit(
        self,
        model_weights,
        test_data,
        reference_data,
        reference_same_atlas_data,
        encoding_ratio,
        task="sdedit",
    ):
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        test_total = 0

        first_batch = True

        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)

                filename = data[4]
                mask_filename = data[5]

                b, _, _, _ = x.shape
                test_total += b

                x_ = self.test_model.sdedit(
                    original_images=x,
                    patho_masks=patho_masks,
                    brain_masks=brain_masks,
                    encoding_ratio=encoding_ratio,
                )

                counterfactuals_np = x_.detach().cpu().numpy()
                predicted_masks = self._predict_segmentation_mask(counterfactuals_np, b)

                # [16,1,128.128]
                if first_batch:
                    all_patho_masks = patho_masks.detach()
                    all_counterfactuals = x_.detach()
                    all_originals = x.detach()
                    all_predicted_masks = predicted_masks
                    first_batch = False
                else:
                    all_patho_masks = torch.cat(
                        (all_patho_masks, patho_masks.detach()), dim=0
                    )
                    all_counterfactuals = torch.cat(
                        (all_counterfactuals, x_.detach()), dim=0
                    )
                    all_originals = torch.cat((all_originals, x.detach()), dim=0)
                    all_predicted_masks.extend(predicted_masks)

                print("shape of all counterfactuals", all_counterfactuals.shape)
                print("shape of all originals", all_originals.shape)

                # print('len of segmentation maps',len(segmentation_maps))

                for batch_idx in range(b):
                    counterfactual = x_[batch_idx].detach().cpu().numpy()
                    #counterfactual[0, 0], counterfactual[0, 1] = 0, 1

                    img = x[batch_idx].detach().cpu().numpy()
                    #img[0, 0], img[0, 1] = 0, 1

                    patho_mask = patho_masks[batch_idx].detach().cpu().numpy()

                    notebook = False
                    if notebook:
                        fig, axs = plt.subplots(1, 4)
                        axs[0].imshow(img[0], cmap="gray")
                        axs[0].set_title("Original Image")
                        axs[0].axis("off")

                        axs[1].imshow(patho_mask[0], cmap="gray")
                        axs[1].set_title("Pathological Mask")
                        axs[1].axis("off")

                        axs[2].imshow(predicted_masks[batch_idx][0], cmap="gray")
                        axs[2].set_title("NNunet predicted mask")
                        axs[2].axis("off")

                        # print('shape of counterfactual[0]', counterfactual[0].shape)
                        axs[3].imshow(counterfactual[0], cmap="gray")
                        axs[3].set_title("Counterfactual")
                        axs[3].axis("off")
                        plt.show()
                    else:
                        grid_image = np.hstack(
                        [img, patho_mask, counterfactual]
                        )
                    
                        grid_all_elements = np.hstack([img, patho_mask, predicted_masks[batch_idx],counterfactual])    

                        index_scan = int(filename[batch_idx].split("_")[-1].replace(".png", ""))
                        index_mask = int(mask_filename[batch_idx].split("_")[-2].replace(".png", ""))

                        wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_pair": [wandb.Image(grid_image)]})

                        wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_cf": [wandb.Image(counterfactual)]})

                        wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_all_elements": [wandb.Image(grid_all_elements)]})

                        wandb.log({task + "/Example_": [wandb.Image(grid_all_elements)]})

            first_batch = True
            with torch.no_grad():
                for data in reference_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_unhealthy = x.detach()
                        first_batch = False
                    else:
                        all_unhealthy = torch.cat((all_unhealthy, x.detach()), dim=0)

            first_batch = True
            with torch.no_grad():
                for data in reference_same_atlas_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_same_atlas = x.detach()
                        first_batch = False
                    else:
                        all_same_atlas = torch.cat((all_same_atlas, x.detach()), dim=0)

            self._compute_metrics(
                all_counterfactuals,
                all_originals,
                all_patho_masks,
                all_unhealthy,
                all_same_atlas,
                all_predicted_masks,
            )

    def repaint(
        self,
        model_weights,
        test_data,
        reference_data,
        reference_same_atlas_data,
        resample_steps,
        task="repaint"
    ):
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        test_total = 0

        first_batch = True
        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)
                dilated_patho_masks = data[3].to(self.device)

                filename = data[4]
                #print('filename', filename)
                mask_filename = data[5]
                #print('mask filename', mask_filename)

                inpaint_masks = dilated_patho_masks

                b, _, _, _ = x.shape
                test_total += b

                #inpaint_masks = patho_masks

                x_ = self.test_model.repaint(
                    original_images=x,
                    inpaint_masks=inpaint_masks,
                    patho_masks=patho_masks,
                    brain_masks=brain_masks,
                    resample_steps=resample_steps,
                )
                counterfactuals_np = x_.detach().cpu().numpy()
                predicted_masks = self._predict_segmentation_mask(counterfactuals_np, b)

                if first_batch:
                    all_patho_masks = patho_masks.detach()
                    all_counterfactuals = x_.detach()
                    all_originals = x.detach()
                    all_predicted_masks = predicted_masks
                    first_batch = False
                else:
                    all_patho_masks = torch.cat(
                        (all_patho_masks, patho_masks.detach()), dim=0
                    )
                    all_counterfactuals = torch.cat(
                        (all_counterfactuals, x_.detach()), dim=0
                    )
                    all_originals = torch.cat((all_originals, x.detach()), dim=0)
                    all_predicted_masks.extend(predicted_masks)

                print("shape of all counterfactuals", all_counterfactuals.shape)
                print("shape of all originals", all_originals.shape)

                for batch_idx in range(b):
                    counterfactual = x_[batch_idx].detach().cpu().numpy()
                    #counterfactual[0, 0], counterfactual[0, 1] = 0, 1

                    img = x[batch_idx].detach().cpu().numpy()
                    #img[0, 0], img[0, 1] = 0, 1

                    patho_mask = patho_masks[batch_idx].detach().cpu().numpy()
                    #patho_mask[0, 0], patho_mask[0, 1] = 0, 1

                    #predicted_mask = predicted_masks[batch_idx]
                    #predicted_mask[0, 0], predicted_mask[0, 1] = 0, 1

                    grid_image = np.hstack(
                        [img, patho_mask, counterfactual]
                    )
                    
                    grid_all_elements = np.hstack([img, patho_mask, predicted_masks[batch_idx],counterfactual])    

                    index_scan = int(filename[batch_idx].split("_")[-1].replace(".png", ""))
                    index_mask = int(mask_filename[batch_idx].split("_")[-2].replace(".png", ""))

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_pair": [wandb.Image(grid_image)]})

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_cf": [wandb.Image(counterfactual)]})

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_all_elements": [wandb.Image(grid_all_elements)]})

                    wandb.log({task + "/Example_": [wandb.Image(grid_all_elements)]})

            first_batch = True
            with torch.no_grad():
                for data in reference_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_unhealthy = x.detach()
                        first_batch = False
                    else:
                        all_unhealthy = torch.cat((all_unhealthy, x.detach()), dim=0)

            first_batch = True
            with torch.no_grad():
                for data in reference_same_atlas_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_same_atlas = x.detach()
                        first_batch = False
                    else:
                        all_same_atlas = torch.cat((all_same_atlas, x.detach()), dim=0)

            self._compute_metrics(
                all_counterfactuals,
                all_originals,
                all_patho_masks,
                all_unhealthy,
                all_same_atlas,
                all_predicted_masks,
            )

    def palette(
        self,
        model_weights,
        test_data,
        reference_data,
        reference_same_atlas_data,
        task="palette",
    ):
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        test_total = 0

        first_batch = True
        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                palette_masks = data[4].to(self.device)

                filename = data[5]
                mask_filename = data[6]

                b, _, _, _ = x.shape
                test_total += b

                x_ = self.test_model.palette(
                    original_images=x, palette_masks=palette_masks
                )

                counterfactuals_np = x_.detach().cpu().numpy()
                predicted_masks = self._predict_segmentation_mask(counterfactuals_np, b)

                if first_batch:
                    all_counterfactuals = x_.detach()
                    all_originals = x.detach()
                    all_palette_masks = palette_masks.detach()
                    all_patho_masks = patho_masks.detach()
                    all_predicted_masks = predicted_masks
                    first_batch = False
                else:
                    all_counterfactuals = torch.cat(
                        (all_counterfactuals, x_.detach()), dim=0
                    )
                    all_originals = torch.cat((all_originals, x.detach()), dim=0)
                    all_palette_masks = torch.cat(
                        (all_palette_masks, palette_masks.detach()), dim=0
                    )
                    all_patho_masks = torch.cat(
                        (all_patho_masks, patho_masks.detach()), dim=0
                    )
                    all_predicted_masks.extend(predicted_masks)

                print("shape of all counterfactuals", all_counterfactuals.shape)
                print("shape of all originals", all_originals.shape)

                for batch_idx in range(b):
                    counterfactual = x_[batch_idx].detach().cpu().numpy()
                    #counterfactual[0, 0], counterfactual[0, 1] = 0, 1

                    img = x[batch_idx].detach().cpu().numpy()
                    #img[0, 0], img[0, 1] = 0, 1

                    patho_mask = patho_masks[batch_idx].detach().cpu().numpy()

                    palette_mask = palette_masks[batch_idx].detach().cpu().numpy()

                    grid_image = np.hstack(
                        [img, patho_mask, counterfactual]
                    )
                    
                    grid_all_elements = np.hstack([img, patho_mask, predicted_masks[batch_idx],counterfactual])    

                    index_scan = int(filename[batch_idx].split("_")[-1].replace(".png", ""))
                    index_mask = int(mask_filename[batch_idx].split("_")[-2].replace(".png", ""))

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_pair": [wandb.Image(grid_image)]})

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_cf": [wandb.Image(counterfactual)]})

                    wandb.log({task + f"/mask_{index_mask}_prior_{index_scan}_all_elements": [wandb.Image(grid_all_elements)]})

                    wandb.log({task + "/Example_": [wandb.Image(grid_all_elements)]})

            first_batch = True
            with torch.no_grad():
                for data in reference_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_unhealthy = x.detach()
                        first_batch = False
                    else:
                        all_unhealthy = torch.cat((all_unhealthy, x.detach()), dim=0)

            first_batch = True
            with torch.no_grad():
                for data in reference_same_atlas_data:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_same_atlas = x.detach()
                        first_batch = False
                    else:
                        all_same_atlas = torch.cat((all_same_atlas, x.detach()), dim=0)

            self._compute_metrics(
                all_counterfactuals,
                all_originals,
                all_patho_masks,
                all_unhealthy,
                all_same_atlas,
                all_predicted_masks,
            )

    def repaint_(self, model_weights, test_data, task="repaint"):

        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + "_loss_rec": 0,
            task + "_loss_mse": 0,
            task + "_loss_pl": 0,
        }
        test_total = 0

        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)
                dilated_patho_masks = data[3].to(self.device)
                inpaint_masks = dilated_patho_masks

                b, _, _, _ = x.shape
                test_total += b

                # expected shape of predicted_x0_source and predicted_x0_target is [b*num_maps_per_mask,1,h,w]
                (
                    inpaint_masks_preliminary,
                    inpaint_masks_non_binarized,
                    predicted_x0_source,
                    predicted_x0_target,
                ) = self.test_model.generate_mask(
                    original_images=x,
                    patho_masks=patho_masks,
                    brain_masks=brain_masks,
                )

                test_mask_self_prediction = False
                if test_mask_self_prediction:
                    for batch_idx in range(b):
                        non_binarized_mask = (
                            inpaint_masks_non_binarized[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        binarized_mask = (
                            inpaint_masks_preliminary[batch_idx].detach().cpu().numpy()
                        )

                        patho_mask = patho_masks[batch_idx].detach().cpu().numpy()
                        contours, _ = cv2.findContours(
                            patho_mask[0].astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        img = x[batch_idx].detach().cpu().numpy()

                        img_rgb = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(img_rgb, contours, -1, (255, 0, 0), 1)

                        binarized_mask_rgb = cv2.cvtColor(
                            binarized_mask[0], cv2.COLOR_GRAY2RGB
                        )
                        non_binarized_mask_rgb = cv2.cvtColor(
                            non_binarized_mask[0], cv2.COLOR_GRAY2RGB
                        )
                        # combine the two binary masks generated mask and patho mask

                        print("shape of binarized mask", binarized_mask_rgb.shape)
                        print(
                            "shape of non binarized mask", non_binarized_mask_rgb.shape
                        )
                        mask_image = np.vstack(
                            [
                                img_rgb * 255,
                                non_binarized_mask_rgb * 255,
                                binarized_mask_rgb * 255,
                            ]
                        )
                        wandb.log({task + "/mask_": [wandb.Image(mask_image)]})
                        # predicted_x0_source is A torch of size [b*num_maps_per_mask,1,h,w], tranform it to a list of
                        # tensors of size [b,1,h,w] with len = num_maps_per_mask
                        predicted_x0_source = predicted_x0_source.split(1, dim=0)
                        predicted_x0_target = predicted_x0_target.split(1, dim=0)

                        # the double slicing is to avoid   ValueError(ValueError: Un-supported shape for image conversion [2, 384, 128]
                        # when using vstack to combine the images. tensor[0][0].detach().cpu().numpy() is a numpy array of size [h,w]
                        # rescale tensor[0][0] to [0,1] from [-1,1] range
                        predicted_x0_source = [
                            (tensor - tensor.min()) / (tensor.max() - tensor.min())
                            for tensor in predicted_x0_source
                        ]
                        predicted_x0_target = [
                            (tensor - tensor.min()) / (tensor.max() - tensor.min())
                            for tensor in predicted_x0_target
                        ]

                        predicted_x0_source_numpy = [
                            tensor[0][0].detach().cpu().numpy()
                            for tensor in predicted_x0_source
                        ]
                        predicted_x0_target_numpy = [
                            tensor[0][0].detach().cpu().numpy()
                            for tensor in predicted_x0_target
                        ]

                        # expected to be [h,w,3]

                        predicted_x0_source_numpy_rgb = [
                            cv2.cvtColor(array, cv2.COLOR_GRAY2RGB) * 255
                            for array in predicted_x0_source_numpy
                        ]
                        predicted_x0_target_numpy_rgb = [
                            cv2.cvtColor(array, cv2.COLOR_GRAY2RGB) * 255
                            for array in predicted_x0_target_numpy
                        ]

                        # show whats inside the predicted_x0_source and predicted_x0_target
                        predicted_x0_source_image = np.vstack(
                            predicted_x0_source_numpy_rgb
                        )
                        predicted_x0_target_image = np.vstack(
                            predicted_x0_target_numpy_rgb
                        )

                        combined_image = np.hstack(
                            [
                                predicted_x0_source_image,
                                predicted_x0_target_image,
                                mask_image,
                            ]
                        )
                        wandb.log(
                            {
                                task
                                + "/source and target x0 predictions": [
                                    wandb.Image(combined_image)
                                ]
                            }
                        )

                        # wandb.log({task + "/predicted_x0_source_images": [wandb.Image(predicted_x0_source_image)]})
                        # wandb.log({task + "/predicted_x0_target_images": [wandb.Image(predicted_x0_target_image)]})

                    continue

                print("shape of patho mask", patho_masks.shape)
                print("shape of inpaint masks trial", inpaint_masks_preliminary.shape)

                # combine the two binary masks masks trial and patho masks
                inpaint_masks = torch.max(inpaint_masks_preliminary, patho_masks)
                # inpaint_masks = brain_masks

                x_ = self.test_model.repaint(
                    original_images=x,
                    inpaint_masks=inpaint_masks,
                    patho_masks=patho_masks,
                    brain_masks=brain_masks,
                )

                print_new_mask_generation_method_results = False
                if print_new_mask_generation_method_results:
                    first_predictions = copy.deepcopy(x_)
                    for i in range(5):
                        x_ = self.test_model.repaint(
                            original_images=x_,
                            inpaint_masks=inpaint_masks,
                            patho_masks=patho_masks,
                            brain_masks=brain_masks,
                        )
                    last_predictions = copy.deepcopy(x_)
                    differential_prediction_maps = torch.abs(
                        last_predictions - first_predictions
                    )

                    clamp_magnitude = differential_prediction_maps.mean() * 6

                    differential_prediction_maps = (
                        differential_prediction_maps.clamp(0, clamp_magnitude)
                        / clamp_magnitude
                    )

                    binarized_differential_prediction_maps = binarize_mask(
                        differential_prediction_maps, 0.7
                    )

                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + "_loss_rec"] += loss_rec.item() * x.size(0)
                metrics[task + "_loss_mse"] += loss_mse.item() * x.size(0)
                metrics[task + "_loss_pl"] += loss_pl.item() * x.size(0)

                for batch_idx in range(b):
                    rec = x_[batch_idx].detach().cpu().numpy()
                    rec[0, 0], rec[0, 1] = 0, 1
                    img = x[batch_idx].detach().cpu().numpy()
                    img[0, 0], img[0, 1] = 0, 1

                    brain_mask = brain_masks[batch_idx].detach().cpu().numpy()
                    patho_mask = patho_masks[batch_idx].detach().cpu().numpy()

                    inpaint_mask = inpaint_masks[batch_idx].detach().cpu().numpy()

                    img_rgb_clean = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)

                    rec_rgb_clean = cv2.cvtColor(rec[0], cv2.COLOR_GRAY2RGB)

                    contours, _ = cv2.findContours(
                        patho_mask[0].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    rec_rgb = cv2.cvtColor(rec[0], cv2.COLOR_GRAY2RGB)
                    cv2.drawContours(rec_rgb, contours, -1, (128, 0, 0), 1)
                    contours_of_changed_area, _ = cv2.findContours(
                        inpaint_mask[0].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(
                        rec_rgb, contours_of_changed_area, -1, (0, 128, 0), 1
                    )

                    img_rgb = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
                    cv2.drawContours(img_rgb, contours, -1, (128, 0, 0), 1)

                    grid_image = np.vstack(
                        [
                            np.hstack([img_rgb_clean * 255, img_rgb * 255]),
                            np.hstack([rec_rgb_clean * 255, rec_rgb * 255]),
                        ]
                    )

                    non_binarized_mask = (
                        inpaint_masks_non_binarized[batch_idx].detach().cpu().numpy()
                    )
                    binarized_mask = (
                        inpaint_masks_preliminary[batch_idx].detach().cpu().numpy()
                    )
                    # combine the two binary masks generated mask and patho mask

                    # print('shape of generated mask',generated_mask.shape)
                    # print('shape of patho mask',patho_mask.shape)
                    mask_image = np.hstack([non_binarized_mask, binarized_mask])
                    wandb.log({task + "/mask_": [wandb.Image(mask_image)]})

                    wandb.log({task + "/Example_": [wandb.Image(grid_image)]})

                    if print_new_mask_generation_method_results:
                        differential_prediction_map = (
                            differential_prediction_maps[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        differential_prediction_map = cv2.cvtColor(
                            differential_prediction_map[0], cv2.COLOR_GRAY2RGB
                        )
                        # wandb.log({task + "/differential_prediction_map": [wandb.Image(differential_prediction_map)]})

                        binarized_differential_prediction_map = (
                            binarized_differential_prediction_maps[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        binarized_differential_prediction_map = cv2.cvtColor(
                            binarized_differential_prediction_map[0], cv2.COLOR_GRAY2RGB
                        )
                        wandb.log(
                            {
                                task
                                + "/binarized_differential_prediction_map": [
                                    wandb.Image(binarized_differential_prediction_map)
                                ]
                            }
                        )

        for metric_key in metrics.keys():
            metric_name = task + "/" + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score})

    def test_mask_self_prediction(self, model_weights, test_data, task="repaint"):
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()

        test_total = 0

        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)
                dilated_patho_masks = data[3].to(self.device)
                inpaint_masks = dilated_patho_masks

                b, _, _, _ = x.shape
                test_total += b

                inpaint_masks_preliminary, inpaint_masks_non_binarized = (
                    self.test_model.generate_mask(
                        original_images=x,
                        patho_masks=patho_masks,
                        brain_masks=brain_masks,
                    )
                )

            for batch_idx in range(b):
                non_binarized_mask = (
                    inpaint_masks_non_binarized[batch_idx]
                    .unsqueeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                # combine the two binary masks generated mask and patho mask

                # print('shape of generated mask',generated_mask.shape)
                # print('shape of patho mask',patho_mask.shape)
                mask_image = np.hstack([non_binarized_mask])

                wandb.log({task + "/mask_": [wandb.Image(mask_image)]})

    def _predict_segmentation_mask(self, counterfactuals_np, batch_size):
        counterfactuals_np = [
            np.expand_dims(counterfactuals_np[i], axis=0) for i in range(batch_size)
        ]

        iterator = self.nnunet.get_data_iterator_from_raw_npy_data(
            counterfactuals_np, None, [PROPS for _ in range(batch_size)], None, 1
        )
        predicted_masks = self.nnunet.predict_from_data_iterator(iterator, False, 1)
        return predicted_masks

    def _compute_metrics(
        self,
        all_counterfactuals,
        all_originals,
        all_patho_masks,
        all_unhealthy,
        all_same_atlas,
        all_predicted_masks,
    ):
        # fid_original_counterfactuals = compute_fid(self.radnet, all_counterfactuals, all_originals)
        # fid_reference_counterfactuals = compute_fid(self.radnet, all_counterfactuals, all_unhealthy)
        fid_reference_same_atlas_counterfactuals, std_fid_same_atlas_counterfactuals = (
            compute_fid(
                self.radnet, all_counterfactuals, all_same_atlas, bootstrap=True,indices_=self.bootstrap_indices
            )
        )

        set_size = all_counterfactuals.shape[0]
        # fid_inception_original_counterfactuals = calculate_fid_given_images([all_originals,all_counterfactuals],set_size,self.device,2048,4)
        # fid_inception_reference_counterfactuals = calculate_fid_given_images([all_unhealthy,all_counterfactuals],set_size,self.device,2048,4)
        fid_inception_reference_same_atlas_counterfactuals, std_fid_inception_same_atlas_counterfactuals = calculate_fid_given_images(
            [all_same_atlas, all_counterfactuals], set_size, self.device, 2048, 4,bootstrap=True, indices_=self.bootstrap_indices
        )

        ssim_original_counterfactuals_mean, ssim_original_counterfactuals_std = (
            compute_ssim(all_counterfactuals, all_patho_masks, all_originals)
        )

        wandb.log(
            {"ssim(original,counterfactuals)_mean": ssim_original_counterfactuals_mean}
        )
        wandb.log(
            {"ssim(original,counterfactuals)_std": ssim_original_counterfactuals_std}
        )

        # msssim_original_counterfactuals = compute_msssim(all_counterfactuals, all_patho_masks, all_originals)
        # wandb.log({"FID Radnet Score (originals,counterfactuals)": fid_original_counterfactuals})
        # wandb.log({"FID Radnet Score (reference_unhealthy,counterfactuals)": fid_reference_counterfactuals})
        wandb.log(
            {
                "FID Radnet Score (reference_same_atlas,counterfactuals)": fid_reference_same_atlas_counterfactuals
            }
        )
        wandb.log(
            {
                "FID Radnet Score (reference_same_atlas,counterfactuals)_std": std_fid_same_atlas_counterfactuals
            }
        )

        # wandb.log({"FID Inception Score (originals,counterfactuals)": fid_inception_original_counterfactuals})
        # wandb.log({"FID Inception Score (reference_unhealthy,counterfactuals)": fid_inception_reference_counterfactuals})
        wandb.log(
            {
                "FID Inception Score (reference_same_atlas,counterfactuals)": fid_inception_reference_same_atlas_counterfactuals
            }
        )

        wandb.log(
            {
                "FID Inception Score (reference_same_atlas,counterfactuals)_std": std_fid_inception_same_atlas_counterfactuals
            }
        )

        all_patho_masks = all_patho_masks.detach().cpu().numpy()
        all_predicted_masks = np.stack(all_predicted_masks)
        print("type of all patho masks", type(all_patho_masks))
        print("shape of all patho masks", all_patho_masks.shape)
        print("shape of predicted patho masks", all_predicted_masks.shape)

        # Compute the Dice coefficient for each batch
        dice = dice_coefficient_batch(all_patho_masks, all_predicted_masks)
        wandb.log({"DICE Score (predicted,conditional) masks": dice})
        print("dice", dice)
