import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import UNet

CHECKPOINT_PATH = "best_model.pth"
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def restore_image(damaged_img: Image.Image, ground_truth_img: Image.Image | None):
    """Runs the model on the uploaded damaged image and returns the restored
    output plus, if a ground-truth image was also provided, PSNR/SSIM text."""
    if damaged_img is None:
        return None, "Please upload a damaged photo first."

    damaged_img = damaged_img.convert("RGB")
    input_tensor = preprocess(damaged_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).clamp(0, 1).squeeze(0).cpu()

    restored_img = transforms.ToPILImage()(output_tensor)

    metrics_text = "Upload a ground-truth image too, to see PSNR / SSIM for this sample."
    if ground_truth_img is not None:
        gt_resized = ground_truth_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        gt_np = np.array(gt_resized).astype(np.float32) / 255.0
        out_np = output_tensor.permute(1, 2, 0).numpy()

        p = psnr(gt_np, out_np, data_range=1.0)
        s = ssim(gt_np, out_np, data_range=1.0, channel_axis=2)
        metrics_text = f"PSNR: {p:.2f} dB   |   SSIM: {s:.3f}"

    return restored_img, metrics_text


with gr.Blocks(title="Old Photo Restoration") as demo:
    gr.Markdown(
        """
        # 🖼️ Old Photo Restoration
        Upload a damaged photo (scratches, noise, fading) and the U-Net model
        will attempt to restore it. Optionally upload the original clean
        photo as well to see PSNR / SSIM for that specific image.
        """
    )

    with gr.Row():
        with gr.Column():
            damaged_input = gr.Image(type="pil", label="Damaged photo (required)")
            gt_input = gr.Image(type="pil", label="Ground truth photo (optional)")
            run_button = gr.Button("Restore", variant="primary")
        with gr.Column():
            restored_output = gr.Image(type="pil", label="Restored output")
            metrics_output = gr.Textbox(label="Metrics", interactive=False)

    run_button.click(
        fn=restore_image,
        inputs=[damaged_input, gt_input],
        outputs=[restored_output, metrics_output],
    )

    gr.Markdown(
        """
        ---
        Model: U-Net (encoder-decoder with skip connections), trained with L1 loss.
        Test-set performance: PSNR ≈ 23.3 dB, SSIM ≈ 0.86.
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
