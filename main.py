"""
TinyGPT Training Script

This script trains a GPT-style language model with configurable attention mechanisms.
Uses DeepSpeed for distributed training and FP16.

Usage:
    deepspeed main.py --attention yat-performer --context-len 1024
"""

import torch
import deepspeed
from torch.utils.data import DataLoader
import json
import os
import time
import wandb

# Import from modular src package
from src import (
    DEFAULT_CONFIG,
    ATTENTION_CLASSES,
    NOVEL_ACTIVATION_TYPES,
    FineWebStream,
    get_eval_loader,
    TinyGPT,
    evaluate,
    log_metrics,
    LossPlateauDetector,
)


# --------------------
# DEEPSPEED CONFIG
# --------------------
def create_ds_config(config):
    """Create DeepSpeed config using experiment parameters."""
    ds_config = {
        "train_micro_batch_size_per_gpu": config['batch_size'],
        "gradient_accumulation_steps": config['gradient_accumulation_steps'],
        "steps_per_print": 100,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config['lr'], 
                "betas": [0.9, 0.95], 
                "eps": 1e-8, 
                "weight_decay": config['weight_decay']
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config['lr'],
                "warmup_num_steps": config.get('warmup_steps', 2000),
                "total_num_steps": config['total_steps']
            }
        }
    }
    
    # Add batch size ramp-up if enabled (DeepSpeed native feature)
    # Format: [start_batch_size, batch_size_increment, rampup_samples]
    # Example: Start at 8 per GPU, ramp to 32 per GPU over rampup_step steps
    if config.get('batch_rampup'):
        start_batch = config.get('batch_rampup_start', 8)  # Start batch per GPU
        final_batch = config['batch_size']  # Final batch per GPU (e.g., 32)
        rampup_steps = config.get('batch_rampup_step', 1000)
        # rampup_samples = steps * batch_size (approximate tokens to ramp over)
        rampup_samples = rampup_steps * start_batch * config['gradient_accumulation_steps']
        batch_increment = max(1, (final_batch - start_batch) // 4)  # Increment in 4 stages
        
        ds_config["train_micro_batch_size_per_gpu"] = final_batch
        ds_config["rampup_batch_size"] = [start_batch, batch_increment, rampup_samples]
    
    with open("ds_zero2.json", "w") as f:
        json.dump(ds_config, f)
    return ds_config


# --------------------
# CLI
# --------------------
def parse_args():
    """Parse command-line arguments for experiment configuration."""
    import argparse
    parser = argparse.ArgumentParser(
        description='TinyGPT with fully configurable experiment parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture
    parser.add_argument('--attention', type=str, default=DEFAULT_CONFIG.get('attention_type', 'performer'),
                        choices=list(ATTENTION_CLASSES.keys()),
                        help='Attention mechanism to use')
    parser.add_argument('--context-len', type=int, default=DEFAULT_CONFIG['context_len'],
                        help='Context window length')
    parser.add_argument('--embed-dim', type=int, default=DEFAULT_CONFIG['embed_dim'],
                        help='Embedding dimension')
    parser.add_argument('--n-layers', type=int, default=DEFAULT_CONFIG['n_layers'],
                        help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=DEFAULT_CONFIG['n_heads'],
                        help='Number of attention heads')
    parser.add_argument('--freeze-embeddings', action='store_true',
                        help='Load pretrained GPT-2 embeddings and freeze them')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Micro batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, 
                        default=DEFAULT_CONFIG['gradient_accumulation_steps'],
                        help='Number of gradient accumulation steps')
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_CONFIG['weight_decay'],
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--total-steps', type=int, default=DEFAULT_CONFIG['total_steps'],
                        help='Total training steps')
    parser.add_argument('--warmup-steps', type=int, default=DEFAULT_CONFIG.get('warmup_steps', 2000),
                        help='Number of warmup steps for LR scheduler')
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG.get('dropout', 0.1),
                        help='Dropout rate for attention and MLP')
    
    # Batch size ramp-up (optional)
    parser.add_argument('--batch-rampup', action='store_true',
                        help='Enable batch size ramp-up (start small, increase to --batch-size)')
    parser.add_argument('--batch-rampup-start', type=int, default=8,
                        help='Starting batch size per GPU (ramps up to --batch-size)')
    parser.add_argument('--batch-rampup-step', type=int, default=1000,
                        help='Number of steps over which to ramp up batch size')
    
    # Logging and checkpointing
    parser.add_argument('--log-dir', type=str, default=DEFAULT_CONFIG['log_dir'],
                        help='Directory for saving logs')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CONFIG['checkpoint_dir'],
                        help='Directory for saving checkpoints')
    parser.add_argument('--eval-interval', type=int, default=DEFAULT_CONFIG['eval_interval'],
                        help='Steps between evaluations')
    parser.add_argument('--save-interval', type=int, default=DEFAULT_CONFIG['save_interval'],
                        help='Steps between checkpoint saves')
    parser.add_argument('--wandb-project', type=str, default=DEFAULT_CONFIG['wandb_project'],
                        help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name for wandb (default: attention_timestamp)')
    
    # Attention-specific parameters
    parser.add_argument('--num-rff-features', type=int, default=DEFAULT_CONFIG['num_rff_features'],
                        help='Number of random Fourier features for RFF attention')
    parser.add_argument('--num-prf-features', type=int, default=DEFAULT_CONFIG['num_prf_features'],
                        help='Number of positive random features for Yat-Performer')
    parser.add_argument('--num-poly-features', type=int, default=DEFAULT_CONFIG['num_poly_features'],
                        help='Number of polynomial features for Yat-Performer')
    parser.add_argument('--num-quadrature-nodes', type=int, default=DEFAULT_CONFIG['num_quadrature_nodes'],
                        help='Number of quadrature nodes for Yat-Performer')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_CONFIG['epsilon'],
                        help='Epsilon for numerical stability in Yat attention')
    
    # DeepSpeed
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (DeepSpeed)')
    
    # Performance optimizations
    parser.add_argument('--use-triton', action='store_true',
                        help='Use Triton-accelerated CUDA kernels for linear attention (36x faster)')
    
    # Data Loading
    parser.add_argument('--num-workers', type=int, default=DEFAULT_CONFIG.get('num_workers', 16),
                        help='Number of data loading workers')
    parser.add_argument('--prefetch-factor', type=int, default=DEFAULT_CONFIG.get('prefetch_factor', 4),
                        help='Number of batches to prefetch per worker')
    
    args, _ = parser.parse_known_args()
    return args


def args_to_config(args):
    """Convert parsed arguments to config dict."""
    config = {
        'context_len': args.context_len,
        'embed_dim': args.embed_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'weight_decay': args.weight_decay,
        'activation_type': DEFAULT_CONFIG['activation_type'],
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'total_steps': args.total_steps,
        'warmup_steps': args.warmup_steps,
        'wandb_project': args.wandb_project,
        'num_rff_features': args.num_rff_features,
        'num_prf_features': args.num_prf_features,
        'num_poly_features': args.num_poly_features,
        'num_quadrature_nodes': args.num_quadrature_nodes,
        'epsilon': args.epsilon,
        'attention_type': args.attention,
        'freeze_embeddings': args.freeze_embeddings,
        'use_wandb': not args.no_wandb,
        'run_name': args.run_name,
        'use_triton': args.use_triton,
        'dropout': args.dropout,
        'batch_rampup': args.batch_rampup,
        'batch_rampup_start': args.batch_rampup_start,
        'batch_rampup_step': args.batch_rampup_step,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
    }
    return config


# --------------------
# MAIN
# --------------------
def main():
    args = parse_args()
    config = args_to_config(args)
    
    ds_config = create_ds_config(config)
    deepspeed.init_distributed()
    rank = deepspeed.comm.get_rank()

    if rank == 0:
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        log_path = os.path.join(config['log_dir'], "training_metrics.csv")
        print(f"Logging metrics to: {log_path}")
        print(f"Using attention type: {config['attention_type']}")
        print(f"Triton acceleration: {'ENABLED' if config.get('use_triton') else 'disabled'}")
        print(f"Config: {json.dumps(config, indent=2, default=str)}")
        
        # Build experiment config for wandb logging
        experiment_config = {
            # Model architecture
            'context_len': config['context_len'],
            'embed_dim': config['embed_dim'],
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'attention_type': config['attention_type'],
            'uses_novel_activation': config['attention_type'] in NOVEL_ACTIVATION_TYPES,
            'freeze_embeddings': config['freeze_embeddings'],
            # Training settings
            'learning_rate': config['lr'],
            'batch_size': config['batch_size'],
            'gradient_accumulation_steps': config['gradient_accumulation_steps'],
            'weight_decay': config['weight_decay'],
            'total_steps': config['total_steps'],
            'eval_interval': config['eval_interval'],
            'save_interval': config['save_interval'],
            'activation_type': config['activation_type'],
            # Attention-specific
            'num_rff_features': config['num_rff_features'],
            'num_prf_features': config['num_prf_features'],
            'num_poly_features': config['num_poly_features'],
            'num_quadrature_nodes': config['num_quadrature_nodes'],
            'epsilon': config['epsilon'],
        }
        
        # Initialize wandb with config
        if config['use_wandb']:
            run_name = config['run_name'] or f"{config['attention_type']}_{int(time.time())}"
            wandb.init(
                project=config['wandb_project'],
                entity="irf-sic",
                name=run_name,
                config=experiment_config
            )
            
            # Define custom x-axes for train loss metrics
            wandb.define_metric("step")
            wandb.define_metric("tokens_processed")
            wandb.define_metric("train_loss_step", step_metric="step")
            wandb.define_metric("train_loss_tokens", step_metric="tokens_processed")

    dataset = FineWebStream(context_len=config['context_len'])
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        config=config,
        attention_type=config['attention_type'],
        freeze_embeddings=config['freeze_embeddings']
    )
    
    # Log model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        if config['use_wandb']:
            wandb.config.update({'num_parameters': num_params, 'vocab_size': dataset.vocab_size})
        print(f"Model parameters: {num_params:,}")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_zero2.json"
    )
    
    # We only load eval data on rank 0 (or all ranks) depending on size
    # Here we load on all but only log on rank 0
    eval_loader = get_eval_loader(config['context_len'], batch_size=16)
    
    loader = DataLoader(
        dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        pin_memory=True,
        num_workers=config['num_workers'],
        prefetch_factor=config['prefetch_factor'],
        )    
    
    if rank == 0:
        print("Starting training loop...")
        if config.get('batch_rampup'):
            print(f"Batch ramp-up enabled (DeepSpeed native): {config['batch_rampup_start']} -> {config['batch_size']} per GPU over {config['batch_rampup_step']} steps")

    # Plateau detector for optimizer switching
    plateau_detector = LossPlateauDetector(patience=3)
    switched_to_sgd = False

    step = 0
    total_tokens = 0
    tokens_per_batch = model_engine.train_micro_batch_size_per_gpu() * config['context_len'] * deepspeed.comm.get_world_size()
    effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps'] * deepspeed.comm.get_world_size()
    t0 = time.time()
    
    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(model_engine.device, dtype=torch.long)
        y = batch[1].to(model_engine.device, dtype=torch.long)
        
        loss = model_engine(x, y)
        model_engine.backward(loss)
        model_engine.step()
        
        step += 1
        total_tokens += tokens_per_batch
        
        # Log training loss with both step and tokens as x-axes
        if rank == 0 and config['use_wandb']:
            # Get current learning rate from DeepSpeed scheduler
            current_lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else config['lr']
            wandb.log({
                "train_loss_step": loss.item(),  # Use step as x-axis
                "train_loss_tokens": loss.item(),  # Use tokens as x-axis
                "learning_rate": current_lr,
                "effective_batch_size": effective_batch_size,
                "step": step,
                "tokens_processed": total_tokens,
            })

        # LOGGING & EVAL
        if step % config['eval_interval'] == 0:
            val_loss, ppl = evaluate(model_engine, eval_loader)
            
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = model_engine.train_micro_batch_size_per_gpu() * config['context_len'] * config['eval_interval'] * deepspeed.comm.get_world_size()
            tps = tokens_processed / dt
            t0 = time.time()
            
            if rank == 0:
                print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | TPS: {tps:.0f}")
                log_metrics(log_path, step, loss.item(), val_loss, ppl, tps)
                if config['use_wandb']:
                    wandb.log({
                        "val_loss": val_loss,
                        "perplexity": ppl,
                        "tokens_per_sec": tps,
                        "step": step
                    })
            
            # Check for plateau and switch to SGD if needed
            if not switched_to_sgd and plateau_detector.check(val_loss):
                if rank == 0:
                    print(f"Step {step} | Loss plateau detected. Switching optimizer from AdamW to SGD.")
                
                switched_to_sgd = True
                
                # Create SGD config based on original config
                sgd_config = ds_config.copy()
                sgd_config['optimizer'] = {
                    "type": "SGD",
                    "params": {
                        "lr": config['lr'],
                        "momentum": 0.9 
                    }
                }
                
                # Write SGD config
                sgd_config_path = "ds_sgd.json"
                with open(sgd_config_path, "w") as f:
                    json.dump(sgd_config, f)
                
                if rank == 0:
                    print("Re-initializing DeepSpeed with SGD...")
                
                # Re-initialize DeepSpeed
                # We reuse the underlying module from the current engine
                current_model = model_engine.module
                model_engine, optimizer, _, _ = deepspeed.initialize(
                    model=current_model,
                    model_parameters=current_model.parameters(),
                    config=sgd_config_path
                )

        # CHECKPOINTING
        if step % config['save_interval'] == 0:
            client_state = {'step': step, 'config': config}
            model_engine.save_checkpoint(config['checkpoint_dir'], tag=f"step_{step}", client_state=client_state)

        if step >= config['total_steps']:
            break

    model_engine.save_checkpoint(config['checkpoint_dir'], tag="final_model", client_state={'config': config})
    if rank == 0:
        print("Training complete. Model saved.")
        if config['use_wandb']:
            print("Uploading model to WandB...")
            artifact = wandb.Artifact(
                name=f"model-{run_name}", 
                type="model",
                description=f"TinyGPT model trained with {config['attention_type']}"
            )
            # Add the DeepSpeed checkpoint subdirectory
            # Note: saved to checkpoint_dir/final_model
            final_model_path = os.path.join(config['checkpoint_dir'], "final_model")
            artifact.add_dir(local_path=final_model_path, name="final_model")
            wandb.log_artifact(artifact)
            wandb.finish()


if __name__ == "__main__":
    main()
