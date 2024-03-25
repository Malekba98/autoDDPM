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


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params["val_interval"]

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

                x_, _ = self.test_model.sample_from_image(
                    x,
                    patho_masks,
                    brain_masks,
                    noise_level=self.model.noise_level_recon,
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

    def repaint(self, model_weights, test_data, task="repaint"):

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

                inpaint_masks_preliminary, inpaint_masks_non_binarized = (
                    self.test_model.generate_mask(
                        original_images=x,
                        patho_masks=patho_masks,
                        brain_masks=brain_masks,
                    )
                )

                test_mask_self_prediction = True
                if test_mask_self_prediction:
                    for batch_idx in range(b):
                        non_binarized_mask = (
                            inpaint_masks_non_binarized[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        binarized_mask = (
                            inpaint_masks_preliminary[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        patho_mask = patho_masks[batch_idx].detach().cpu().numpy()
                        contours, _ = cv2.findContours(
                        patho_mask[0].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                        )
                        img = x[batch_idx].detach().cpu().numpy()
                        img_rgb = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(img_rgb, contours, -1, (128, 0, 0), 1)

                        binarized_mask_rgb = cv2.cvtColor(binarized_mask[0], cv2.COLOR_GRAY2RGB)
                        non_binarized_mask_rgb = cv2.cvtColor(non_binarized_mask[0], cv2.COLOR_GRAY2RGB)
                        # combine the two binary masks generated mask and patho mask

                        print("shape of binarized mask", binarized_mask.shape)
                        print("shape of non binarized mask", non_binarized_mask.shape)
                        mask_image = np.vstack([img_rgb*255,non_binarized_mask_rgb*255, binarized_mask_rgb*255])
                        wandb.log({task + "/mask_": [wandb.Image(mask_image)]})
                    continue

                print("shape of patho mask", patho_masks.shape)
                print("shape of inpaint masks trial", inpaint_masks_preliminary.shape)

                # combine the two binary masks masks trial and patho masks
                inpaint_masks = torch.max(inpaint_masks_preliminary, patho_masks)

                x_ = self.test_model.repaint(
                    original_images=x,
                    inpaint_masks=inpaint_masks,
                    patho_masks=patho_masks,
                    brain_masks=brain_masks,
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
                        inpaint_masks_non_binarized[batch_idx]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    binarized_mask = (
                            inpaint_masks_preliminary[batch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    # combine the two binary masks generated mask and patho mask

                    # print('shape of generated mask',generated_mask.shape)
                    # print('shape of patho mask',patho_mask.shape)
                    mask_image = np.hstack([non_binarized_mask, binarized_mask])
                    wandb.log({task + "/mask_": [wandb.Image(mask_image)]})

                    wandb.log({task + "/Example_": [wandb.Image(grid_image)]})

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
