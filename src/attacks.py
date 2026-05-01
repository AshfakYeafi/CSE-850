import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, steps, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    adv_images = original_images + torch.empty_like(original_images).uniform_(
        -epsilon, epsilon
    )
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        perturbation = torch.clamp(
            adv_images - original_images,
            min=-epsilon,
            max=epsilon,
        )
        adv_images = torch.clamp(original_images + perturbation, 0.0, 1.0)

    return adv_images.detach()
