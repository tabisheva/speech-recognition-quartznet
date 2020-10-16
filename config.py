model_config = [
    {'filters': 256, 'repeat': 1, 'kernel': 33, 'stride': 2, 'dilation': 1, 'residual': False, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': 33, 'stride': 1, 'dilation': 1, 'residual': True, 'separable': True},
    {'filters': 256, 'repeat': 5, 'kernel': 39, 'stride': 1, 'dilation': 1, 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': 51, 'stride': 1, 'dilation': 1, 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': 63, 'stride': 1, 'dilation': 1, 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 5, 'kernel': 75, 'stride': 1, 'dilation': 1, 'residual': True, 'separable': True},
    {'filters': 512, 'repeat': 1, 'kernel': 87, 'stride': 1, 'dilation': 2, 'residual': False, 'separable': True},
    {'filters': 1024, 'repeat': 1, 'kernel': 1, 'stride': 1, 'dilation': 1, 'residual': False, 'separable': False}
]
params = {"num_features": 64,
          "sample_rate": 16000,
          "original_sample_rate": 22050,
          "batch_size": 128,
          "num_workers": 8,
          "lr": 0.0005,
          "num_epochs": 100,
          "noise_variance": 0.005,
          "min_time_stretch": 0.9,
          "max_time_stretch": 1.1,
          "min_shift": -3,
          "max_shift": 3,
          "time_masking": 5,
          "wandb_name": "Quartznet_LJSpeech",
          "clip_grad_norm": 15,
          }