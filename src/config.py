"""
Default configuration for TinyGPT training.
"""

DEFAULT_CONFIG = {
    'context_len': 1024,
    'embed_dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'lr': 3e-4,
    'batch_size': 64,  # Optimized for A100 (40GB/80GB)
    'gradient_accumulation_steps': 1,  # Global batch = 64 * 8 * 1 = 512
    'weight_decay': 0.01,
    'activation_type': 'novel',
    'log_dir': 'logs_fineweb_bert',
    'checkpoint_dir': 'checkpoints_fineweb_bert',
    'eval_interval': 500,
    'save_interval': 1000,
    'total_steps': 20000,  # Increased for more significant training (~10B tokens)
    'warmup_steps': 2000,  # LR warmup steps (~10% of total_steps)
    'wandb_project': 'SLAY_Ablation_350M',
    'num_rff_features': 64,
    'num_prf_features': 16,
    'num_poly_features': 8,
    'num_quadrature_nodes': 2,
    'epsilon': 0.001,
    'use_triton': False,  # Use Triton-accelerated CUDA kernels for linear attention
    'dropout': 0.1,  # Dropout rate for attention and MLP
    'batch_rampup': False,  # Enable batch size ramp-up
    'batch_rampup_start': 8,  # Starting batch size per GPU
    'batch_rampup_step': 1000,  # Steps over which to ramp up
    'num_workers': 4,  # Reduced from 16 to prevent excessive thread spawn (4*8=32 total)
    'prefetch_factor': 4,  # Prefetch batches to avoid GPU starvation
}
