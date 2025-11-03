import torch
import os
from omegaconf import OmegaConf
from einops import rearrange
import sys
import importlib

sys.path.append("../Generation")
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from collections import OrderedDict


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Extrapolator:
    def __init__(self, args):
        self.args = args
        self.model = self.create_model()

    def create_model(self):
        diffusion_ckpt_path = self.args.diffusion_ckpt
        diffusion_config_path = self.args.diffusion_config
        config = OmegaConf.load(diffusion_config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        assert os.path.exists(diffusion_ckpt_path), "Error: checkpoint Not Found!"
        # load ckpt
        state_dict = torch.load(diffusion_ckpt_path, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                ## rename the keys for 256x256 model
                new_pl_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_pl_sd[k] = v

                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed", "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                model.load_state_dict(new_pl_sd, strict=True)
        else:
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict["module"].keys():
                new_pl_sd[key[16:]] = state_dict["module"][key]
            model.load_state_dict(new_pl_sd)
        print(">>> model checkpoint loaded.")
        # set model to eval mode
        model.eval()
        model.perframe_ae = True
        model = model.half()
        model = model.cuda()
        return model

    def get_latent_z(self, videos):
        with torch.no_grad(), torch.cuda.amp.autocast():
            z = self.model.encode_first_stage(videos)

        return z

    def image_guided_synthesis(
        self,
        rgb_video,
        depth_video,
        ref_frames,
        noise_shape,
        init_rgb=None,
        init_depth=None,
        n_samples=1,
        ddim_steps=50,
        ddim_eta=1.0,
        unconditional_guidance_scale=1.0,
        cfg_img=None,
        fs=None,
        text_input=False,
        multiple_cond_cfg=False,
        loop=False,
        interp=False,
        timestep_spacing="uniform",
        guidance_rescale=0.0,
        **kwargs,
    ):
        ddim_sampler = (
            DDIMSampler(self.model)
            if not multiple_cond_cfg
            else DDIMSampler_multicond(self.model)
        )

        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.model.device)

        # img = videos[:,:,0] #bchw
        # img_emb = model.embedder(img) ## blc
        # img_emb = model.image_proj_model(img_emb)
        with torch.no_grad(), torch.cuda.amp.autocast():
            ref_frames = rearrange(ref_frames, "b c l h w -> (b l) c h w")
            img_emb = self.model.embedder(ref_frames)  ## (b l) c
            img_emb = self.model.image_proj_model(img_emb)
            img_emb = rearrange(img_emb, "(b t) l c -> b (t l) c", b=1)
            cond = {"c_crossattn": [torch.cat([img_emb], dim=1)]}

            if self.model.model.conditioning_key == "hybrid":
                videos = torch.cat([rgb_video, depth_video], dim=1)

                img_cat_cond = self.get_latent_z(videos).half().detach()  # b c t h w
                cond["c_concat"] = [img_cat_cond]  # b c 1 h w

            init_latent_z = None

            if init_rgb is not None and init_depth is not None:
                init_latent_z = self.get_latent_z(
                    torch.cat([init_rgb, init_depth], dim=1)
                )

            if unconditional_guidance_scale != 1.0:
                uc_img_emb = self.model.embedder(torch.zeros_like(ref_frames))  ## b l c
                uc_img_emb = self.model.image_proj_model(uc_img_emb)
                uc_img_emb = rearrange(uc_img_emb, "(b t) l c -> b (t l) c", b=1)
                uc = {"c_crossattn": [torch.cat([uc_img_emb], dim=1)]}
                if self.model.model.conditioning_key == "hybrid":
                    uc["c_concat"] = [img_cat_cond]
            else:
                uc = None

            kwargs.update({"unconditional_conditioning_img_nonetext": None})

            # z0 = img_cat_cond#None
            z0 = None  # None
            cond_mask = None

            results = []
            for _ in range(n_samples):
                if z0 is not None:
                    cond_z0 = z0.clone()
                    kwargs.update({"clean_cond": True})
                else:
                    cond_z0 = None
                if ddim_sampler is not None:
                    # x_t = ddim_sampler.stochastic_encode(img_cat_cond, ddim_steps,use_original_steps=True

                    x_t = init_latent_z

                    samples, _ = ddim_sampler.sample(
                        S=ddim_steps,
                        conditioning=cond,
                        batch_size=batch_size,
                        shape=noise_shape[1:],
                        verbose=False,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        cfg_img=cfg_img,
                        mask=cond_mask,
                        x0=cond_z0,
                        x_T=x_t,
                        fs=fs,
                        timestep_spacing=timestep_spacing,
                        precision=16,
                        guidance_rescale=guidance_rescale,
                        **kwargs,
                    )

                batch_images = self.model.decode_first_stage(samples, return_depth=True)

                results.append(batch_images)

            results = torch.stack(results)
            return results.permute(1, 0, 2, 3, 4, 5)

    def repair(
        self, artifact_rgb, artifact_depth, ref_frames, init_rgb=None, init_depth=None
    ):
        with torch.no_grad(), torch.cuda.amp.autocast():
            n_frames = artifact_rgb.shape[1]

            _, n_frames, h, w = artifact_rgb.shape  # (3, n_frames, 320, 512)
            h = h // 16 * 2
            w = w // 16 * 2

            artifact_rgb = artifact_rgb.unsqueeze(0).cuda()
            artifact_depth = artifact_depth.unsqueeze(0).cuda()
            ref_frames = ref_frames.unsqueeze(0).cuda()
            artifact_rgb = artifact_rgb.half()
            artifact_depth = artifact_depth.half()
            ref_frames = ref_frames.half()
            if init_rgb is not None and init_depth is not None:
                init_rgb = init_rgb.unsqueeze(0).cuda().half()
                init_depth = init_depth.unsqueeze(0).cuda().half()

            noise_shape = [1, 4, n_frames, h, w]

            batch_samples = self.image_guided_synthesis(
                artifact_rgb,
                artifact_depth,
                ref_frames,
                noise_shape,
                init_rgb,
                init_depth,
                n_samples=1,
                ddim_steps=25,
                ddim_eta=1.0,
                unconditional_guidance_scale=self.args.unconditional_guidance_scale,
                cfg_img=None,
                fs=30,
                guidance_rescale=0.7,
            )
            batch_samples = batch_samples[0, 0]
            repaired_rgb = batch_samples[:3]
            repaired_depth = batch_samples[3:]
            repaired_rgb = torch.clamp(repaired_rgb, -1.0, 1.0)
            orig_repaired_rgb = repaired_rgb.detach().clone()
            repaired_rgb = (repaired_rgb + 1.0) / 2.0
            repaired_depth = torch.clamp(repaired_depth, -1.0, 1.0)
            orig_repaired_depth = repaired_depth
            repaired_depth = (repaired_depth + 1.0) / 2.0
            repaired_depth = 1 / (repaired_depth + 1e-6)
            return repaired_rgb, repaired_depth, orig_repaired_rgb, orig_repaired_depth

    def depth_normalize(self, depths):
        epsilon = 1e-10
        valid_mask = depths > 0
        disparity = torch.zeros_like(depths)
        disparity[valid_mask] = 1.0 / (depths[valid_mask] + epsilon)
        valid_disparities = torch.masked_select(disparity, valid_mask)

        if valid_disparities.numel() > 0:
            disp_min = valid_disparities.min()
            disp_max = valid_disparities.max()
            normalized_disparity = torch.zeros_like(disparity)
            normalized_disparity[valid_mask] = (disparity[valid_mask] - disp_min) / (
                disp_max - disp_min
            )
            print(
                "normalized_disparity:",
                normalized_disparity.max(),
                normalized_disparity.min(),
            )

        else:
            print("Warning: No valid depth values found")
            normalized_disparity = torch.zeros_like(disparity)

        return normalized_disparity
