# """
# Train multiple models in parallel using multiple GPUs or processes.
# This significantly reduces total training time.
# """

# import os
# import json
# import argparse
# import torch
# import multiprocessing as mp
# from train_models import main as train_single

# def train_model_wrapper(args_dict):
#     """Wrapper function for parallel training."""
#     import sys

#     # Convert dict back to argparse-like object
#     class Args:
#         def __init__(self, d):
#             for key, value in d.items():
#                 setattr(self, key, value)

#     args = Args(args_dict)

#     # Override sys.argv for the train script
#     sys.argv = ['train_models.py',
#                 '--data_dir', args.data_dir,
#                 '--output_dir', args.output_dir,
#                 '--batch_size', str(args.batch_size),
#                 '--num_epochs', str(args.num_epochs),
#                 '--learning_rate', str(args.learning_rate),
#                 '--num_workers', str(args.num_workers),
#                 '--models', args.models]

#     if args.no_amp:
#         sys.argv.append('--no_amp')

#     # Run training
#     train_single()


# def main():
#     parser = argparse.ArgumentParser(description='Train models in parallel')
#     parser.add_argument('--data_dir', type=str, default='./data/processed')
#     parser.add_argument('--output_dir', type=str, default='./')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--num_epochs', type=int, default=20)
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--no_amp', action='store_true')
#     args = parser.parse_args()

#     # Check GPU availability
#     num_gpus = torch.cuda.device_count()

#     print(f"{'='*60}")
#     print(f"PARALLEL TRAINING SETUP")
#     print(f"{'='*60}")
#     print(f"Available GPUs: {num_gpus}")

#     if num_gpus == 0:
#         print("\nWARNING: No GPUs available. Falling back to sequential training.")
#         print("Parallel training requires at least 1 GPU.")
#         return

#     # Configure training jobs
#     if num_gpus >= 2:
#         # With 2+ GPUs: Train ResNet on GPU 0, ViT on GPU 1 simultaneously
#         print("\nStrategy: Training ResNet and ViT in parallel on separate GPUs")
#         print("  GPU 0: ResNet Multi-task")
#         print("  GPU 1: ViT Multi-task")
#         print("\nThis will cut training time in HALF!")

#         jobs = [
#             {
#                 'data_dir': args.data_dir,
#                 'output_dir': args.output_dir,
#                 'batch_size': args.batch_size,
#                 'num_epochs': args.num_epochs,
#                 'learning_rate': args.learning_rate,
#                 'num_workers': args.num_workers // 2,  # Split workers
#                 'models': 'resnet',
#                 'no_amp': args.no_amp,
#                 'gpu_id': 0
#             },
#             {
#                 'data_dir': args.data_dir,
#                 'output_dir': args.output_dir,
#                 'batch_size': args.batch_size,
#                 'num_epochs': args.num_epochs,
#                 'learning_rate': args.learning_rate,
#                 'num_workers': args.num_workers // 2,
#                 'models': 'vit',
#                 'no_amp': args.no_amp,
#                 'gpu_id': 1
#             }
#         ]

#         # Set GPU affinity and launch processes
#         processes = []
#         for job in jobs:
#             # Set environment variable for GPU
#             env = os.environ.copy()
#             env['CUDA_VISIBLE_DEVICES'] = str(job['gpu_id'])

#             # Create process
#             p = mp.Process(target=train_model_wrapper, args=(job,))
#             p.start()
#             processes.append(p)

#         # Wait for all to complete
#         for p in processes:
#             p.join()

#     else:
#         # With 1 GPU: Can't truly parallelize, but can optimize
#         print("\nStrategy: Sequential training on single GPU")
#         print("  Training ResNet first, then ViT")
#         print("\nNote: To train in parallel, you need 2+ GPUs")
#         print("Current approach will take ~2-5 hours")

#         # Train sequentially
#         print("\n" + "="*60)
#         print("Training ResNet Multi-task...")
#         print("="*60)
#         os.system(f"python train_models.py --models resnet --batch_size {args.batch_size} "
#                  f"--num_epochs {args.num_epochs} --num_workers {args.num_workers}")

#         print("\n" + "="*60)
#         print("Training ViT Multi-task...")
#         print("="*60)
#         os.system(f"python train_models.py --models vit --batch_size {args.batch_size} "
#                  f"--num_epochs {args.num_epochs} --num_workers {args.num_workers}")

#     print("\n" + "="*60)
#     print("PARALLEL TRAINING COMPLETE!")
#     print("="*60)


# if __name__ == '__main__':
#     # Required for multiprocessing on some systems
#     mp.set_start_method('spawn', force=True)
#     main()
