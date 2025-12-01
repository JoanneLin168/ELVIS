import torch

def get_random_coordinates(image, patch_size):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    h, w = image.shape[-2:]
    x1 = torch.randint(0, w - patch_size[1] + 1, (1,)).item()
    y1 = torch.randint(0, h - patch_size[0] + 1, (1,)).item()
    x2 = x1 + patch_size[1]
    y2 = y1 + patch_size[0]

    return (x1, y1, x2, y2)

def crop(image, coordinates):
    x1, y1, x2, y2 = coordinates
    return image[..., y1:y2, x1:x2]

def crop_image(image, patch_size, coordinates=None, random=False):
    if coordinates is not None:
        return crop(image, coordinates)
    
    if random:
        coodinates = get_random_coordinates(image, patch_size)
        image = crop(image, coodinates)
    else:
        height, width = image.shape[-2:]
        coordinates = (
            (width - patch_size) // 2,
            (height - patch_size) // 2,
            (width + patch_size) // 2,
            (height + patch_size) // 2
        )

    return crop(image, coordinates)
        