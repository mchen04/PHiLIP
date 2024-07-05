# import torch
# from PIL import Image
# from torchvision import transforms
# from VAR.models.basic_vae import Encoder, Decoder  # Assuming these are from your VQVAE setup
# # from VAR.models import basic_vae

# def load_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     return img

# def process_images(image_paths, vae_model, var_model, device):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     images = []
#     for path in image_paths:
#         img = load_image(path)
#         img_tensor = transform(img).unsqueeze(0).to(device)
#         images.append(img_tensor)

#     batch = torch.cat(images, dim=0)
#     with torch.no_grad():
#         # Encode images using VQVAE
#         latent_vectors = vae_model.encode(batch)
        
#         # Process latent vectors using VAR model
#         concepts = var_model.autoregressive_infer_cfg(B=len(latent_vectors), label_B=latent_vectors, cfg=4, top_k=900, top_p=0.95)
    
#     return concepts

# def generate_final_image(concepts, prompt, temperature):
#     # Decode the concepts to get the final high-resolution image
#     final_image_tensor = vae.decode(concepts)
#     final_image = transforms.ToPILImage()(final_image_tensor.cpu().squeeze(0))
#     return final_image

import torch
from PIL import Image
from torchvision import transforms
from VAR.models.basic_vae import Encoder, Decoder  # Adjust paths as necessary
from VAR.models.basic_var import VAR  # Assuming VAR is implemented similarly

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def process_images(image_paths, vae_model, var_model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []
    for path in image_paths:
        img = load_image(path)
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append(img_tensor)

    batch = torch.cat(images, dim=0)
    with torch.no_grad():
        # Encode images using VQVAE
        latent_vectors = vae_model.encode(batch)
        
        # Process latent vectors using VAR model
        concepts = var_model.autoregressive_infer_cfg(B=len(latent_vectors), label_B=latent_vectors, cfg=4, top_k=900, top_p=0.95)
    
    return concepts

def generate_final_image(concepts, vae_model, device):
    with torch.no_grad():
        final_image_tensor = vae_model.decode(concepts).cpu()
        final_image = transforms.ToPILImage()(final_image_tensor.squeeze(0))
    return final_image

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    vae_encoder = Encoder(ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.0, in_channels=3, z_channels=256)
    vae_decoder = Decoder(ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.0, in_channels=3, z_channels=256)
    vae_encoder.load_state_dict(torch.load('vae_ch160v4096z32.pth'))
    vae_decoder.load_state_dict(torch.load('vae_ch160v4096z32.pth'))
    vae_encoder.to(device)
    vae_decoder.to(device)
    
    var_model = VAR(...)  # Initialize VAR model appropriately
    var_model.load_state_dict(torch.load('var_d16.pth'))
    var_model.to(device)

    # Test image
    image_paths = ['generated_images/512-selected-1']

    # Process images
    concepts = process_images(image_paths, vae_encoder, var_model, device)

    # Generate final image
    final_image = generate_final_image(concepts, vae_decoder, device)
    final_image.save('output_image.jpg')
    final_image.show()

if __name__ == '__main__':
    main()
