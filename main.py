"""
Pilot test script for spatiotemporal video prediction using Moving MNIST.
Includes reproducibility tests and ablation studies.
Fixed version that handles custom metric serialization and mixed precision issues.
"""

import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns


def configure_gpu():
    """Configure GPU settings for optimal performance."""
    print("Configuring GPU settings...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    if gpus:
        try:
            # Enable memory growth to prevent GPU memory allocation errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set the first GPU as the default device
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Check if AVX-512 is available before enabling mixed precision
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            has_avx512 = 'avx512' in str(cpu_info.get('flags', '')).lower()
            
            if has_avx512:
                # Enable mixed precision for better GPU performance only if AVX-512 is available
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision enabled (AVX-512 detected)")
            else:
                print("AVX-512 not detected, keeping float32 precision for compatibility")
            
            print(f"GPU configured successfully: {gpus[0]}")
            
            # Print GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Details: {gpu_details}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
        except ImportError:
            print("cpuinfo not available, disabling mixed precision for safety")
            tf.keras.mixed_precision.set_global_policy('float32')
    else:
        print("No GPU found. Running on CPU.")
        print("WARNING: Training will be significantly slower on CPU")
        # Ensure float32 policy for CPU
        tf.keras.mixed_precision.set_global_policy('float32')
    
    return len(gpus) > 0


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess Moving MNIST dataset."""
    print("Downloading and loading Moving MNIST dataset...")
    
    # Download and load dataset
    fpath = keras.utils.get_file(
        "moving_mnist.npy",
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    )
    dataset = np.load(fpath)
    
    # Preprocess dataset
    dataset = np.swapaxes(dataset, 0, 1)
    dataset = dataset[:1000, ...]
    dataset = np.expand_dims(dataset, axis=-1)
    
    # Split into train and validation
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    
    # Normalize
    train_dataset = train_dataset / 255.0
    val_dataset = val_dataset / 255.0
    
    print(f"Dataset loaded: Train {train_dataset.shape}, Val {val_dataset.shape}")
    
    return train_dataset, val_dataset, train_index, val_index


def create_shifted_frames(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create shifted frames for temporal prediction."""
    x = data[:, 0 : data.shape[1] - 1, :, :, :]  # Keep all 5 dimensions
    y = data[:, 1 : data.shape[1], :, :, :]      # Keep all 5 dimensions
    return x, y


# Custom metrics - registered for serialization with improved handling
@keras.saving.register_keras_serializable()
def sequence_ce_sum(y_true, y_pred):
    """Paper-style summed cross-entropy per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Get the actual number of dimensions to handle both 4D and 5D cases
    ndims = len(bce.shape)
    if ndims == 4:  # (batch, time, height, width)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3, 4])
    return tf.reduce_mean(ce_sum_per_seq)


@keras.saving.register_keras_serializable()
def mse_seq(y_true, y_pred):
    """MSE per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    se = tf.square(y_pred - y_true)
    # Get the actual number of dimensions
    ndims = len(se.shape)
    if ndims == 4:  # (batch, time, height, width)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3, 4])
    return tf.reduce_mean(mse_per_seq)


@keras.saving.register_keras_serializable()
def ssim_seq(y_true, y_pred):
    """SSIM per sequence with improved mixed precision handling."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Get shape using tf.shape to handle dynamic shapes properly
    shape = tf.shape(y_true)
    batch_size = shape[0]
    time_steps = shape[1]
    height = shape[2]
    width = shape[3]
    
    # Handle both 4D and 5D cases
    if len(y_true.shape) == 4:  # (batch, time, height, width)
        channels = 1
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    else:  # (batch, time, height, width, channels)
        channels = shape[4]
    
    # Reshape for SSIM computation - ensure we have proper float32 tensors
    y_true_reshaped = tf.reshape(y_true, (batch_size * time_steps, height, width, channels))
    y_pred_reshaped = tf.reshape(y_pred, (batch_size * time_steps, height, width, channels))
    
    # Ensure inputs are float32
    y_true_reshaped = tf.cast(y_true_reshaped, tf.float32)
    y_pred_reshaped = tf.cast(y_pred_reshaped, tf.float32)
    
    # Compute SSIM
    frame_ssim = tf.image.ssim(y_true_reshaped, y_pred_reshaped, max_val=1.0)
    frame_ssim = tf.reshape(frame_ssim, (batch_size, time_steps))
    seq_ssim = tf.reduce_mean(frame_ssim, axis=1)
    return tf.reduce_mean(seq_ssim)


class SimpleBaseline(layers.Layer):
    """Simple baseline layer that copies last frame to all output timesteps."""
    
    def call(self, inputs):
        # Take the last frame and repeat it for all timesteps
        last_frame = inputs[:, -1:, :, :, :]  # Shape: (batch, 1, H, W, C)
        time_steps = tf.shape(inputs)[1]
        repeated = tf.repeat(last_frame, repeats=time_steps, axis=1)
        return repeated
    
    def compute_output_shape(self, input_shape):
        return input_shape


def create_naive_baseline_model(input_shape: Tuple) -> keras.Model:
    """Create naive baseline model that simply copies the last input frame to all outputs."""
    inp = layers.Input(shape=input_shape)
    output = SimpleBaseline()(inp)
    return keras.Model(inp, output, name="Naive_FrameCopy")


def create_baseline_model(input_shape: Tuple) -> keras.Model:
    """Create baseline model with 3 ConvLSTM2D layers."""
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=input_shape)
    
    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)
    
    return keras.Model(inp, x, name="Baseline_3ConvLSTM")


def create_ablated_model(input_shape: Tuple) -> keras.Model:
    """Create ablated model with 2 ConvLSTM2D layers (one less than baseline)."""
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=input_shape)
    
    # 2 ConvLSTM2D layers with batch normalization,
    # followed by a Conv3D layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)
    
    return keras.Model(inp, x, name="Ablated_2ConvLSTM")


def train_model(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray, 
                x_val: np.ndarray, y_val: np.ndarray, model_name: str, run: int) -> Dict:
    """Train a model and return results."""
    
    print(f"\n=== Training {model_name} - Run {run} ===")
    print(f"Model parameters: {model.count_params():,}")
    
    # Use GPU device context
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Compile model with mixed precision considerations
        optimizer = keras.optimizers.Adam()
        
        # For mixed precision, wrap optimizer
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("Using mixed precision with loss scaling")
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.BinaryCrossentropy(from_logits=False, name="bce_avg"),
                sequence_ce_sum,
                mse_seq,
                ssim_seq,
            ],
        )
        
        # Callbacks for short test run
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, verbose=1, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=1, verbose=1, factor=0.5
        )
        
        # Add GPU memory monitoring callback
        class GPUMemoryCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 1 == 0:  # Check every epoch for short runs
                    if tf.config.list_physical_devices('GPU'):
                        try:
                            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                            print(f"Epoch {epoch}: GPU Memory - Current: {gpu_info['current']//1024//1024}MB, "
                                  f"Peak: {gpu_info['peak']//1024//1024}MB")
                        except:
                            pass  # Skip if memory info unavailable
        
        gpu_callback = GPUMemoryCallback()
        
        # Train model
        print(f"Starting training for {model_name} - Run {run}")
        print(f"Device: {'/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'}")
        print("WARNING: Running with only 3 epochs for testing - increase for full experiments")
        
        history = model.fit(
            x_train, y_train,
            batch_size=5,
            epochs=3,  # Short test run
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr, gpu_callback],
            verbose=2  # Less verbose output
        )
        
        # Get final metrics
        print(f"Evaluating final performance for {model_name} - Run {run}")
        final_metrics = model.evaluate(x_val, y_val, verbose=0)
    
    # Extract metric names
    metric_names = model.metrics_names
    
    # Debug: Print metric names to understand what was compiled
    print(f"Model metric names: {metric_names}")
    print(f"Final metrics values: {final_metrics}")
    
    results = {
        'model_name': model_name,
        'run': run,
        'seed': [42, 123, 456, 789, 101112, 131415, 222324][run - 1],
        'model_params': model.count_params(),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'best_val_loss': min(history.history['val_loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': final_metrics[0],
    }
    
    # Add all final metrics with explicit mapping
    for i, metric_name in enumerate(metric_names[1:], 1):  # Skip loss (already added)
        clean_metric_name = metric_name.replace('_1', '').replace('_2', '').replace('_3', '')
        results[f'final_val_{clean_metric_name}'] = final_metrics[i]
        print(f"  Added metric: final_val_{clean_metric_name} = {final_metrics[i]:.6f}")
    
    # Add training history statistics
    results.update({
        'min_train_loss': min(history.history['loss']),
        'min_val_loss': min(history.history['val_loss']),
        'final_learning_rate': float(model.optimizer.learning_rate.numpy()),
    })
    
    # Add SSIM-specific tracking from history if available
    if 'val_ssim_seq' in history.history:
        results['max_val_ssim_seq'] = max(history.history['val_ssim_seq'])
    elif any('ssim' in key for key in history.history.keys()):
        # Find the SSIM key in history
        ssim_key = next(key for key in history.history.keys() if 'ssim' in key and 'val' in key)
        results['max_val_ssim_seq'] = max(history.history[ssim_key])
    
    print(f"Results for {model_name} Run {run}:")
    for key, value in results.items():
        if isinstance(value, (int, float)) and key not in ['model_name', 'run', 'seed']:
            print(f"  {key}: {value:.6f}")
    
    return results, history.history


def calculate_significance_tests(results_df: pd.DataFrame, ssim_column: str) -> Dict:
    """Calculate statistical significance tests between models."""
    try:
        from scipy import stats
        
        significance_results = {}
        models = results_df['model_name'].unique()
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                model1_ssim = results_df[results_df['model_name'] == model1][ssim_column]
                model2_ssim = results_df[results_df['model_name'] == model2][ssim_column]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(model1_ssim, model2_ssim)
                
                significance_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'model1_mean': model1_ssim.mean(),
                    'model2_mean': model2_ssim.mean(),
                }
        
        return significance_results
    except ImportError:
        print("scipy not available for significance testing")
        return {}


def analyze_metric_correlations(results_df: pd.DataFrame) -> Dict:
    """Analyze correlations between different metrics for convergent validity."""
    
    # Find available validation metrics
    metric_cols = [col for col in results_df.columns if col.startswith('final_val_') and col != 'final_val_loss']
    
    if len(metric_cols) < 2:
        return {'error': 'Not enough metrics for correlation analysis'}
    
    correlation_results = {}
    
    # Calculate correlation matrix
    corr_matrix = results_df[metric_cols].corr()
    correlation_results['correlation_matrix'] = corr_matrix.to_dict()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Metric Correlation Matrix (Convergent Validity)')
    plt.tight_layout()
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    corr_plot_file = f'metric_correlations_{timestamp}.png'
    plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    correlation_results['plot_saved'] = corr_plot_file
    
    # Analyze specific correlations
    correlation_results['analysis'] = {}
    
    # Check if SSIM correlates negatively with loss metrics (expected)
    ssim_cols = [col for col in metric_cols if 'ssim' in col.lower()]
    loss_cols = [col for col in metric_cols if any(term in col.lower() for term in ['mse', 'bce', 'ce'])]
    
    for ssim_col in ssim_cols:
        for loss_col in loss_cols:
            if ssim_col in corr_matrix.columns and loss_col in corr_matrix.columns:
                corr_val = corr_matrix.loc[ssim_col, loss_col]
                correlation_results['analysis'][f'{ssim_col}_vs_{loss_col}'] = {
                    'correlation': float(corr_val),
                    'expected_negative': corr_val < 0,
                    'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate' if abs(corr_val) > 0.3 else 'weak'
                }
    
    # Overall convergent validity assessment
    avg_abs_corr = corr_matrix.abs().mean().mean()
    correlation_results['convergent_validity'] = {
        'average_absolute_correlation': float(avg_abs_corr),
        'assessment': 'good' if avg_abs_corr > 0.6 else 'moderate' if avg_abs_corr > 0.3 else 'poor'
    }
    
    return correlation_results


def generate_visual_inspection_samples(models_dict: Dict, x_val: np.ndarray, y_val: np.ndarray, 
                                     n_samples: int = 5) -> Dict:
    """Generate visual samples for qualitative face validity assessment."""
    
    visual_results = {}
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Select random samples for visualization
    np.random.seed(42)  # Fixed seed for consistent sample selection
    sample_indices = np.random.choice(len(x_val), n_samples, replace=False)
    
    for model_name, model in models_dict.items():
        print(f"Generating visual samples for {model_name}...")
        
        try:
            # Get predictions for sample indices
            sample_x = x_val[sample_indices]
            sample_y = y_val[sample_indices]
            
            # Generate predictions with explicit float32 casting
            with tf.device('/CPU:0'):  # Force CPU for inference to avoid mixed precision issues
                predictions = model.predict(sample_x.astype(np.float32), verbose=0)
            
            # Ensure predictions are float32
            predictions = predictions.astype(np.float32)
            
            # Create visualization
            fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
            fig.suptitle(f'Visual Inspection: {model_name}', fontsize=16, y=0.98)
            
            for i in range(n_samples):
                # Input sequence (show first and last frame)
                axes[i, 0].imshow(sample_x[i, 0, :, :, 0], cmap='gray')
                axes[i, 0].set_title(f'Sample {i+1}: Input (t=0)')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(sample_x[i, -1, :, :, 0], cmap='gray')
                axes[i, 1].set_title(f'Input (t={sample_x.shape[1]-1})')
                axes[i, 1].axis('off')
                
                # Ground truth (show last frame)
                axes[i, 2].imshow(sample_y[i, -1, :, :, 0], cmap='gray')
                axes[i, 2].set_title(f'Ground Truth (t={sample_y.shape[1]})')
                axes[i, 2].axis('off')
                
                # Prediction (show last frame)
                axes[i, 3].imshow(predictions[i, -1, :, :, 0], cmap='gray')
                axes[i, 3].set_title(f'Prediction (t={sample_y.shape[1]})')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            visual_file = f'visual_inspection_{model_name}_{timestamp}.png'
            plt.savefig(visual_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate sample-specific metrics for qualitative assessment
            sample_metrics = {}
            for i in range(n_samples):
                pred_frame = predictions[i:i+1, -1:, :, :, :].astype(np.float32)
                true_frame = sample_y[i:i+1, -1:, :, :, :].astype(np.float32)
                
                # Calculate SSIM for this sample using CPU to avoid mixed precision issues
                with tf.device('/CPU:0'):
                    ssim_val = tf.image.ssim(
                        tf.reshape(tf.cast(true_frame, tf.float32), (1, *true_frame.shape[2:4], 1)),
                        tf.reshape(tf.cast(pred_frame, tf.float32), (1, *pred_frame.shape[2:4], 1)),
                        max_val=1.0
                    ).numpy()[0]
                
                # Calculate MSE for this sample
                mse_val = np.mean((pred_frame - true_frame) ** 2)
                
                sample_metrics[f'sample_{i+1}'] = {
                    'ssim': float(ssim_val),
                    'mse': float(mse_val),
                    'qualitative_assessment': 'good' if ssim_val > 0.8 else 'moderate' if ssim_val > 0.6 else 'poor'
                }
            
            visual_results[model_name] = {
                'visualization_file': visual_file,
                'sample_metrics': sample_metrics,
                'average_sample_ssim': float(np.mean([metrics['ssim'] for metrics in sample_metrics.values()])),
                'average_sample_mse': float(np.mean([metrics['mse'] for metrics in sample_metrics.values()]))
            }
            
            print(f"  Generated: {visual_file}")
            
        except Exception as e:
            print(f"  Failed to generate visual samples for {model_name}: {e}")
            visual_results[model_name] = {
                'error': str(e),
                'visualization_file': None,
                'sample_metrics': {},
                'average_sample_ssim': None,
                'average_sample_mse': None
            }
    
    return visual_results


def save_model_weights_and_predictions(models_dict: Dict, x_val: np.ndarray, timestamp: str) -> Dict:
    """Alternative approach to store model information without cloning."""
    
    model_storage = {}
    
    for model_name, model in models_dict.items():
        print(f"Saving weights and generating predictions for {model_name}...")
        
        try:
            # Save model weights with correct filename format
            weights_file = f'{model_name}_weights_{timestamp}.weights.h5'  # Fixed extension
            model.save_weights(weights_file)
            
            # Generate some predictions for analysis
            with tf.device('/CPU:0'):  # Force CPU inference to avoid mixed precision issues
                sample_predictions = model.predict(x_val[:10].astype(np.float32), verbose=0)
            
            model_storage[model_name] = {
                'weights_file': weights_file,
                'model_config': model.get_config(),
                'sample_predictions_shape': sample_predictions.shape,
                'sample_predictions_mean': float(np.mean(sample_predictions)),
                'sample_predictions_std': float(np.std(sample_predictions)),
                'architecture_summary': {
                    'total_params': model.count_params(),
                    'trainable_params': model.count_params(),
                    'layers': len(model.layers)
                }
            }
            
            print(f"  Saved weights: {weights_file}")
            print(f"  Generated sample predictions: {sample_predictions.shape}")
            
        except Exception as e:
            print(f"  Failed to save {model_name}: {e}")
            model_storage[model_name] = {
                'error': str(e),
                'weights_file': None,
                'sample_predictions_shape': None,
                'sample_predictions_mean': None,
                'sample_predictions_std': None,
                'architecture_summary': {
                    'total_params': model.count_params(),
                    'trainable_params': model.count_params(),
                    'layers': len(model.layers)
                }
            }
    
    return model_storage


def main():
    """Main function to run all experiments."""
    print("Starting Pilot Test for Spatiotemporal Video Prediction")
    print("=" * 70)
    
    # Configure GPU first
    gpu_available = configure_gpu()
    print("=" * 70)
    
    print("This script tests:")
    print("1. Reproducibility across multiple random seeds (reliability)")
    print("2. Baseline comparisons including naive frame-copying (criterion validity)")
    print("3. Ablation study (2 vs 3 ConvLSTM layers) (construct validity)")
    print("4. Metric correlation analysis (convergent validity)")
    print("5. Visual inspection of predictions (face validity)")
    print("=" * 70)
    
    # Load data once
    print("\n" + "="*50)
    print("DATA LOADING AND PREPROCESSING")
    print("="*50)
    
    train_dataset, val_dataset, train_idx, val_idx = load_and_preprocess_data()
    
    # Create shifted frames
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    
    print(f"Final data shapes:")
    print(f"  X_train: {x_train.shape}")
    print(f"  Y_train: {y_train.shape}")
    print(f"  X_val: {x_val.shape}")
    print(f"  Y_val: {y_val.shape}")
    
    # Define input shape
    input_shape = (None, *x_train.shape[2:])
    print(f"  Model input shape: {input_shape}")
    
    # Store all results
    all_results = []
    all_histories = {}
    trained_models = {}  # Store trained models for visual inspection
    
    # Define models to test
    model_configs = [
        ('Naive_FrameCopy', create_naive_baseline_model),
        ('Baseline_3ConvLSTM', create_baseline_model),
        ('Ablated_2ConvLSTM', create_ablated_model),
    ]
    
    # Run experiments with more seeds for better statistical power
    seeds = [42, 123, 456, 789, 101112, 131415, 222324]  # 7 seeds for robust testing
    
    print(f"\n" + "="*50)
    print("EXPERIMENTAL RUNS")
    print("="*50)
    print(f"Testing {len(model_configs)} models with {len(seeds)} different seeds each")
    print(f"Total experiments: {len(model_configs) * len(seeds)}")
    
    for model_idx, (model_name, model_creator) in enumerate(model_configs, 1):
        print(f"\n{'='*70}")
        print(f"TESTING MODEL {model_idx}/{len(model_configs)}: {model_name}")
        print(f"{'='*70}")
        
        for run, seed in enumerate(seeds, 1):
            print(f"\n{'-'*50}")
            print(f"Run {run}/{len(seeds)} with seed {seed}")
            print(f"{'-'*50}")
            
            # Set seed for reproducibility
            set_seeds(seed)
            
            # Create model
            model = model_creator(input_shape)
            
            # Train model and collect results
            results, history = train_model(
                model, x_train, y_train, x_val, y_val, model_name, run
            )
            all_results.append(results)
            all_histories[f"{model_name}_run_{run}"] = history
            
            # Store the best model from the first run of each model type for visual inspection
            # Use a different approach that doesn't require cloning
            if run == 1:
                # Create a fresh model with the same architecture
                fresh_model = model_creator(input_shape)
                # Copy weights from trained model
                fresh_model.set_weights(model.get_weights())
                trained_models[model_name] = fresh_model
            
            # Clear model to free memory
            del model
            keras.backend.clear_session()
            tf.keras.utils.clear_session()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Debug: Print actual column names to understand what metrics were saved
    print(f"\nActual DataFrame columns:")
    for col in sorted(results_df.columns):
        print(f"  {col}")
    
    print(f"\n{'='*70}")
    print("PILOT TEST SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    # Find available metrics dynamically
    available_metrics = [col for col in results_df.columns if col.startswith('final_val_')]
    base_metrics = ['final_val_loss', 'epochs_trained', 'best_val_loss']
    
    # Combine base metrics with available validation metrics
    metrics_to_analyze = base_metrics + available_metrics
    # Remove duplicates while preserving order
    metrics_to_analyze = list(dict.fromkeys(metrics_to_analyze))
    
    print(f"\nAnalyzing metrics: {metrics_to_analyze}")
    
    # Calculate comprehensive statistics
    stats_df = results_df.groupby('model_name')[metrics_to_analyze].agg({
        metric: ['mean', 'std', 'min', 'max'] for metric in metrics_to_analyze
    }).round(6)
    
    print("\nDetailed Statistics:")
    print(stats_df)
    
    # Perform metric correlation analysis for convergent validity
    print(f"\n{'='*50}")
    print("METRIC CORRELATION ANALYSIS (CONVERGENT VALIDITY)")
    print(f"{'='*50}")
    
    try:
        correlation_results = analyze_metric_correlations(results_df)
        if 'error' not in correlation_results:
            print(f"Correlation analysis completed. Heatmap saved: {correlation_results['plot_saved']}")
            print(f"Average absolute correlation: {correlation_results['convergent_validity']['average_absolute_correlation']:.3f}")
            print(f"Convergent validity assessment: {correlation_results['convergent_validity']['assessment']}")
            
            if correlation_results['analysis']:
                print("\nKey metric relationships:")
                for relationship, data in correlation_results['analysis'].items():
                    print(f"  {relationship}: {data['correlation']:.3f} ({'expected' if data['expected_negative'] else 'unexpected'}, {data['strength']})")
        else:
            print(f"Correlation analysis failed: {correlation_results['error']}")
            correlation_results = {}
    except Exception as e:
        print(f"Correlation analysis failed: {e}")
        correlation_results = {}
    
    # Generate visual inspection samples for face validity
    print(f"\n{'='*50}")
    print("VISUAL INSPECTION (FACE VALIDITY)")
    print(f"{'='*50}")
    
    try:
        visual_results = generate_visual_inspection_samples(trained_models, x_val, y_val, n_samples=5)
        print("Visual inspection samples generated:")
        for model_name, data in visual_results.items():
            if 'error' not in data:
                print(f"  {model_name}: {data['visualization_file']}")
                print(f"    Average sample SSIM: {data['average_sample_ssim']:.3f}")
                print(f"    Average sample MSE: {data['average_sample_mse']:.6f}")
            else:
                print(f"  {model_name}: Failed - {data['error']}")
    except Exception as e:
        print(f"Visual inspection generation failed: {e}")
        visual_results = {}
    
    # Calculate statistical significance using available SSIM metric
    ssim_column = None
    for col in results_df.columns:
        if 'ssim' in col.lower() and col.startswith('final_val_'):
            ssim_column = col
            break
    
    significance_results = {}
    if ssim_column:
        try:
            significance_results = calculate_significance_tests(results_df, ssim_column)
            print(f"\n{'='*50}")
            print("STATISTICAL SIGNIFICANCE TESTS")
            print(f"{'='*50}")
            
            for comparison, results in significance_results.items():
                print(f"\n{comparison}:")
                print(f"  t-statistic: {results['t_statistic']:.4f}")
                print(f"  p-value: {results['p_value']:.4f}")
                print(f"  Significant (α=0.05): {results['significant']}")
                print(f"  Mean SSIM difference: {abs(results['model1_mean'] - results['model2_mean']):.4f}")
        except Exception as e:
            print(f"Statistical significance testing failed: {e}")
    else:
        print(f"\nNo SSIM metric found for significance testing")
    
    # Save model information using alternative approach
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        model_storage = save_model_weights_and_predictions(trained_models, x_val, timestamp)
        print(f"\n{'='*50}")
        print("MODEL STORAGE COMPLETED")
        print(f"{'='*50}")
        
        for model_name, storage_info in model_storage.items():
            if 'error' not in storage_info:
                print(f"{model_name}:")
                print(f"  Weights saved: {storage_info['weights_file']}")
                print(f"  Total parameters: {storage_info['architecture_summary']['total_params']:,}")
                print(f"  Prediction sample mean: {storage_info['sample_predictions_mean']:.4f}")
            else:
                print(f"{model_name}: Failed - {storage_info['error']}")
    except Exception as e:
        print(f"Model storage failed: {e}")
        model_storage = {}
    
    # Prepare comprehensive results for JSON export
    comprehensive_results = {
        'experiment_metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'gpu_available': gpu_available,
            'total_experiments': len(all_results),
            'models_tested': list(results_df['model_name'].unique()),
            'seeds_used': seeds,
            'epochs_per_run': 3,
            'data_shapes': {
                'x_train': list(x_train.shape),
                'y_train': list(y_train.shape),
                'x_val': list(x_val.shape),
                'y_val': list(y_val.shape),
                'input_shape': list(input_shape)
            },
            'mixed_precision_policy': str(tf.keras.mixed_precision.global_policy().name)
        },
        'individual_runs': results_df.to_dict('records'),
        'training_histories': all_histories,
        'summary_statistics': {},
        'significance_tests': significance_results,
        'metric_correlations': correlation_results,
        'visual_inspection': visual_results,
        'model_storage': model_storage,
        'reliability_assessment': {},
        'ablation_analysis': {},
        'validity_assessment': {},
        'key_findings': {}
    }
    
    # Convert summary statistics to JSON-friendly format
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        comprehensive_results['summary_statistics'][model_name] = {}
        
        for metric in metrics_to_analyze:
            if metric in model_data.columns:
                comprehensive_results['summary_statistics'][model_name][metric] = {
                    'mean': float(model_data[metric].mean()),
                    'std': float(model_data[metric].std()),
                    'min': float(model_data[metric].min()),
                    'max': float(model_data[metric].max())
                }
    
    # Add reliability assessment to JSON
    if ssim_column:
        for model_name in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model_name]
            ssim_std = model_data[ssim_column].std()
            ssim_mean = model_data[ssim_column].mean()
            cv = ssim_std / ssim_mean if ssim_mean != 0 else float('inf')
            
            comprehensive_results['reliability_assessment'][model_name] = {
                'ssim_mean': float(ssim_mean),
                'ssim_std': float(ssim_std),
                'coefficient_of_variation': float(cv),
                'reliability_rating': 'High' if cv < 0.1 else 'Moderate' if cv < 0.2 else 'Low',
                'reproducibility_score': float(max(0, 1 - cv))  # Higher is better, bounded at 0
            }
    
    # Add ablation analysis to JSON - WITH ERROR HANDLING
    ablation_analysis = {}
    if ssim_column:
        baseline_data = results_df[results_df['model_name'] == 'Baseline_3ConvLSTM']
        ablated_data = results_df[results_df['model_name'] == 'Ablated_2ConvLSTM']
        
        if len(baseline_data) > 0 and len(ablated_data) > 0:
            baseline_ssim = baseline_data[ssim_column].mean()
            ablated_ssim = ablated_data[ssim_column].mean()
            performance_difference = ((baseline_ssim - ablated_ssim) / ablated_ssim * 100) if ablated_ssim != 0 else 0
            
            ablation_analysis = {
                'baseline_3convlstm_ssim': float(baseline_ssim),
                'ablated_2convlstm_ssim': float(ablated_ssim),
                'performance_difference_percent': float(performance_difference),
                'third_layer_beneficial': performance_difference > 0,
                'effect_size': abs(performance_difference),
                'conclusion': "The third ConvLSTM layer contributes positively to model performance" if performance_difference > 0 else "The third ConvLSTM layer does not improve model performance"
            }
    
    comprehensive_results['ablation_analysis'] = ablation_analysis
    
    # Add comprehensive validity assessment to JSON
    validity_assessment = {
        'face_validity': {},
        'criterion_validity': {},
        'construct_validity': {},
        'convergent_validity': {}
    }
    
    # Face validity from visual inspection
    if visual_results:
        validity_assessment['face_validity'] = {
            'visual_samples_generated': True,
            'models_inspected': list(visual_results.keys()),
            'average_visual_quality': {
                model: data.get('average_sample_ssim', None) for model, data in visual_results.items()
            },
            'assessment': 'Visual inspection samples generated for qualitative evaluation'
        }
    
    # Criterion validity from baseline comparisons
    if ssim_column:
        naive_data = results_df[results_df['model_name'] == 'Naive_FrameCopy']
        baseline_data = results_df[results_df['model_name'] == 'Baseline_3ConvLSTM']
        if len(naive_data) > 0 and len(baseline_data) > 0:
            naive_ssim = naive_data[ssim_column].mean()
            baseline_ssim = baseline_data[ssim_column].mean()
            
            improvement_over_naive = ((baseline_ssim - naive_ssim) / naive_ssim * 100) if naive_ssim != 0 else 0
            
            validity_assessment['criterion_validity'] = {
                'naive_baseline_ssim': float(naive_ssim),
                'model_baseline_ssim': float(baseline_ssim),
                'improvement_over_naive_percent': float(improvement_over_naive),
                'substantially_better': improvement_over_naive > 10,  # >10% improvement threshold
                'assessment': f"Model shows {improvement_over_naive:.1f}% improvement over naive baseline"
            }
    
    # Construct validity from ablation study
    if ablation_analysis:  # Check if ablation_analysis exists and is not empty
        validity_assessment['construct_validity'] = {
            'ablation_study_conducted': True,
            'architecture_components_tested': ['third_convlstm_layer'],
            'component_contribution_significant': ablation_analysis.get('third_layer_beneficial', False),
            'assessment': ablation_analysis.get('conclusion', 'Ablation analysis completed')
        }
    
    # Convergent validity from metric correlations
    if correlation_results and 'convergent_validity' in correlation_results:
        validity_assessment['convergent_validity'] = correlation_results['convergent_validity']
        validity_assessment['convergent_validity']['assessment'] = f"Convergent validity: {correlation_results['convergent_validity']['assessment']}"
    
    comprehensive_results['validity_assessment'] = validity_assessment
    
    # Add key findings summary
    comprehensive_results['key_findings'] = {
        'best_performing_model': None,
        'most_reliable_model': None,
        'parameter_efficiency': {},
        'performance_summary': {}
    }
    
    # Find best performing model
    if ssim_column and len(results_df) > 0:
        model_means = results_df.groupby('model_name')[ssim_column].mean()
        best_model_name = model_means.idxmax()
        best_model_ssim = model_means.max()
        
        comprehensive_results['key_findings']['best_performing_model'] = {
            'model_name': best_model_name,
            'mean_ssim': float(best_model_ssim)
        }
        
        # Find most reliable model (lowest coefficient of variation)
        reliability_scores = {}
        for model_name in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model_name]
            ssim_mean = model_data[ssim_column].mean()
            ssim_std = model_data[ssim_column].std()
            cv = ssim_std / ssim_mean if ssim_mean != 0 else float('inf')
            reliability_scores[model_name] = cv
        
        most_reliable = min(reliability_scores.keys(), key=lambda x: reliability_scores[x])
        comprehensive_results['key_findings']['most_reliable_model'] = {
            'model_name': most_reliable,
            'coefficient_of_variation': float(reliability_scores[most_reliable])
        }
    else:
        comprehensive_results['key_findings']['best_performing_model'] = {
            'model_name': 'Unknown',
            'mean_ssim': None,
            'note': 'No SSIM metric available for comparison'
        }
        comprehensive_results['key_findings']['most_reliable_model'] = {
            'model_name': 'Unknown',
            'coefficient_of_variation': None,
            'note': 'No SSIM metric available for reliability assessment'
        }
    
    # Add parameter efficiency analysis
    loss_col = 'final_val_loss'
    mse_col = next((col for col in results_df.columns if 'mse' in col.lower() and col.startswith('final_val_')), None)
    
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        params = model_data['model_params'].iloc[0]
        
        performance_metrics = {}
        if ssim_column:
            performance_metrics['ssim'] = float(model_data[ssim_column].mean())
        if loss_col in model_data.columns:
            performance_metrics['validation_loss'] = float(model_data[loss_col].mean())
        if mse_col:
            performance_metrics['mse'] = float(model_data[mse_col].mean())
        
        comprehensive_results['key_findings']['parameter_efficiency'][model_name] = {
            'parameters': int(params),
            'performance_metrics': performance_metrics,
            'parameters_per_ssim_point': float(params / performance_metrics.get('ssim', 1)) if 'ssim' in performance_metrics and performance_metrics['ssim'] > 0 else None
        }
    
    # Performance summary for each model
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        
        summary = {
            'parameters': int(model_data['model_params'].iloc[0]),
            'epochs_to_convergence': {
                'mean': float(model_data['epochs_trained'].mean()),
                'std': float(model_data['epochs_trained'].std())
            }
        }
        
        if ssim_column:
            summary['ssim'] = {
                'mean': float(model_data[ssim_column].mean()),
                'std': float(model_data[ssim_column].std())
            }
        if loss_col in model_data.columns:
            summary['validation_loss'] = {
                'mean': float(model_data[loss_col].mean()),
                'std': float(model_data[loss_col].std())
            }
        if mse_col:
            summary['mse'] = {
                'mean': float(model_data[mse_col].mean()),
                'std': float(model_data[mse_col].std())
            }
        
        comprehensive_results['key_findings']['performance_summary'][model_name] = summary
    
    # Save comprehensive results to JSON
    import json
    json_file = f'pilot_test_comprehensive_results_{timestamp}.json'
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    try:
        with open(json_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nComprehensive results saved to: {json_file}")
    except Exception as e:
        print(f"Failed to save comprehensive results: {e}")
    
    # Print key findings for dissertation
    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR DISSERTATION")
    print(f"{'='*70}")
    
    # Find key metrics dynamically
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        
        print(f"\n{model_name}:")
        print(f"  Parameters: {model_data['model_params'].iloc[0]:,}")
        
        if ssim_column:
            print(f"  SSIM (Mean ± Std): {model_data[ssim_column].mean():.4f} ± {model_data[ssim_column].std():.4f}")
        if loss_col in model_data.columns:
            print(f"  Validation Loss: {model_data[loss_col].mean():.4f} ± {model_data[loss_col].std():.4f}")
        if mse_col:
            print(f"  MSE: {model_data[mse_col].mean():.6f} ± {model_data[mse_col].std():.6f}")
        
        print(f"  Epochs to convergence: {model_data['epochs_trained'].mean():.1f} ± {model_data['epochs_trained'].std():.1f}")
        
        if ssim_column:
            print(f"  Reproducibility (SSIM std): {model_data[ssim_column].std():.4f}")
    
    print(f"\n{'='*70}")
    print("PILOT TEST COMPLETED SUCCESSFULLY")
    print(f"Total runtime: Training completed for {len(all_results)} experiments")
    print(f"Mixed precision policy used: {tf.keras.mixed_precision.global_policy().name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()