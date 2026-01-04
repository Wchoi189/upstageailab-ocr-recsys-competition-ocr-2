
import timm
import torch

def check_vit():
    model_name = "vit_small_patch16_224"
    print(f"Creating model {model_name}...")
    try:
        model = timm.create_model(model_name, pretrained=False, features_only=True)
        print("Success with features_only=True")
    except RuntimeError as e:
        print(f"Failed with features_only=True: {e}")
        model = timm.create_model(model_name, pretrained=False, features_only=False)
        print("Success with features_only=False")

    x = torch.randn(1, 3, 224, 224)
    out = model.forward_features(x)
    print(f"forward_features output type: {type(out)}")
    if isinstance(out, torch.Tensor):
        print(f"forward_features output shape: {out.shape}")

    print("Model properties:")
    print(f"embed_dim: {getattr(model, 'embed_dim', 'N/A')}")
    print(f"num_features: {getattr(model, 'num_features', 'N/A')}")
    print(f"feature_info: {getattr(model, 'feature_info', 'N/A')}")

if __name__ == "__main__":
    check_vit()
