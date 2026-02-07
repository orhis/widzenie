from torchvision import transforms


def get_transforms(variant: str, image_size: int = 224):
    """
    Zwraca transforms dla train oraz test.
    variant: "A1" | "A2" | "A3" | "A4"
    image_size: docelowy rozmiar wejścia (domyślnie 224)
    """

    # TEST zawsze identyczny (zero augmentacji)
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # A1: baseline (kontrola) — tylko resize + normalize
    if variant == "A1":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, test_tf

    # A2: geometry — flip + rotation (+ delikatne affine)
    if variant == "A2":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, test_tf

    # A3: photometric — jitter + lekkie rozmycie
    if variant == "A3":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, test_tf

    # A4: domain gap / regularization — grayscale + erasing (redukcja nadmiernej zależności od tekstur)
    if variant == "A4":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, test_tf

    raise ValueError(f"Nieznany wariant augmentacji: {variant}. Użyj A1/A2/A3/A4.")
