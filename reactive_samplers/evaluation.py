"""Post-generation quality scorers using the project's downloaded reference checkpoints."""

from pathlib import Path
import sys


class QualityMetrics:
    def __init__(self, names, clip_model=None, image_reward_checkpoint=None, image_reward_config=None):
        unknown = set(names) - {"clip_score", "image_reward", "psnr", "ssim", "lpips"}
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}")
        self.names = names
        if "lpips" in names:
            import lpips
            import torch
            self.lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lpips = lpips.LPIPS(net="alex", version="0.1").to(self.lpips_device).eval()
        root = Path(__file__).resolve().parents[1]
        if "image_reward" in names:
            checkpoint = Path(image_reward_checkpoint or root / "metrics/image_reward/checkpoints/ImageReward.pt")
            config = Path(image_reward_config or checkpoint.parent / "med_config.json")
            for path in (checkpoint, config):
                if not path.is_file():
                    raise FileNotFoundError(path)
            import ImageReward as RM
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.image_reward = RM.load(str(checkpoint), device=device, med_config=str(config))
            self.image_reward.eval()
        if "clip_score" in names:
            if not clip_model:
                raise ValueError("metrics.clip_model must name the CLIP checkpoint")
            checkpoint = Path(clip_model)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sys.path.insert(0, str(root / "metrics/clip/source"))
            import clip
            import torch
            self.clip, self.clip_preprocess = clip.load(str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu")
            self.clip.eval()
            self.clip_tokenize = clip.tokenize

    def score(self, image_path, prompt, reference_image=None):
        scores = {}
        pair_names = set(self.names) & {"psnr", "ssim", "lpips"}
        if pair_names:
            if reference_image is None:
                raise ValueError("Paired metrics require a matching baseline image")
            import numpy as np
            from PIL import Image
            with Image.open(image_path) as im:
                generated = np.array(im.convert("RGB"))
            with Image.open(reference_image) as im:
                reference = np.array(im.convert("RGB"))
            if generated.shape != reference.shape:
                raise ValueError("Generated and baseline images must have the same dimensions")
            if pair_names & {"psnr", "ssim"}:
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                if "psnr" in pair_names:
                    # Identical images have infinite PSNR, which JSON cannot represent.
                    scores["psnr"] = None if np.array_equal(reference, generated) else float(peak_signal_noise_ratio(reference, generated, data_range=255))
                if "ssim" in pair_names:
                    scores["ssim"] = float(structural_similarity(reference, generated, data_range=255, channel_axis=2))
            if "lpips" in pair_names:
                import torch
                def tensor(image):
                    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.lpips_device, dtype=torch.float32) / 127.5 - 1
                with torch.inference_mode():
                    scores["lpips"] = float(self.lpips(tensor(reference), tensor(generated)).item())
        if "image_reward" in self.names:
            import torch
            with torch.inference_mode():
                scores["image_reward"] = float(self.image_reward.score(prompt, str(image_path)))
        if "clip_score" in self.names:
            import torch
            from PIL import Image
            with Image.open(image_path) as im:
                image = self.clip_preprocess(im.convert("RGB")).unsqueeze(0).to(next(self.clip.parameters()).device)
            tokens = self.clip_tokenize([prompt], truncate=True).to(image.device)
            with torch.inference_mode():
                image_features = self.clip.encode_image(image).float()
                text_features = self.clip.encode_text(tokens).float()
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
            scores["clip_score"] = max(100.0 * similarity, 0.0)
        return scores


def paired_feedback(result, references):
    """Compare only matching prompt/seed outputs; scores are separate signals, not percentages."""
    reference = next((row for row in references
                      if (row["prompt"], row["seed"]) == (result["prompt"], result["seed"])), None)
    if reference is None:
        return {}
    reference_scores = {name: reference.get("scores", {})[name] for name in result["scores"]
                        if name in {"clip_score", "image_reward"} and name in reference.get("scores", {})}
    return {"reference_image": reference["image"], "reference_scores": reference_scores,
            "score_deltas": {name: result["scores"][name] - score for name, score in reference_scores.items()},
            "reference_nfe": reference["nfe"], "nfe_reduction": 1 - result["nfe"] / reference["nfe"],
            "reference_valid_for_benchmark": reference.get("valid_for_benchmark", True)}
