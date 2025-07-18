import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle as pk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


class ModelSaver:
    """Helper class for saving models, metrics, and visualizations."""

    def __init__(self, experiment_name, base_dir="experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir

        # Create timestamp for unique experiment folder
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")

        # Create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "metrics"), exist_ok=True)

        # Initialize metric storage
        self.metrics_history = {
            'train_loss': [], 'valid_loss': [], 'test_loss': [],
            'train_acc_emo': [], 'valid_acc_emo': [], 'test_acc_emo': [],
            'train_f1_emo': [], 'valid_f1_emo': [], 'test_f1_emo': [],
            'train_acc_sen': [], 'valid_acc_sen': [], 'test_acc_sen': [],
            'train_f1_sen': [], 'valid_f1_sen': [], 'test_f1_sen': [],
            'train_acc_sft': [], 'valid_acc_sft': [], 'test_acc_sft': [],
            'train_f1_sft': [], 'valid_f1_sft': [], 'test_f1_sft': [],
            'epoch': []
        }

        self.best_metrics = {}
        self.confusion_matrices = {}

        print(f"Experiment directory created: {self.exp_dir}")

    def update_metrics(self, epoch=None, **metrics):
        """Update metrics for the current epoch."""
        # If epoch is provided, ensure we track it
        if epoch is not None:
            if not self.metrics_history['epoch'] or self.metrics_history['epoch'][-1] != epoch:
                self.metrics_history['epoch'].append(epoch)
                # Initialize all metrics lists for this epoch with None
                for key in self.metrics_history:
                    if key != 'epoch' and len(self.metrics_history[key]) < len(self.metrics_history['epoch']):
                        self.metrics_history[key].append(None)

        # Update provided metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                if epoch is not None:
                    # Replace the last entry (current epoch)
                    if self.metrics_history[key] and len(self.metrics_history[key]) == len(
                            self.metrics_history['epoch']):
                        self.metrics_history[key][-1] = value
                    else:
                        self.metrics_history[key].append(value)
                else:
                    # Just update the last entry without adding epoch
                    if self.metrics_history[key]:
                        self.metrics_history[key][-1] = value
                    else:
                        self.metrics_history[key].append(value)

    def save_model(self, model, optimizer, epoch, is_best=False, extra_info=None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp
        }

        if extra_info:
            checkpoint.update(extra_info)

        # Save best model
        if is_best:
            best_path = os.path.join(self.exp_dir, "models", "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")

    def save_confusion_matrix(self, y_true, y_pred, labels, title, task_type="emotion"):
        """Save confusion matrix as plot and CSV."""
        cm = confusion_matrix(y_true, y_pred)

        # Save as CSV
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        csv_path = os.path.join(self.exp_dir, "metrics", f"confusion_matrix_{task_type}.csv")
        cm_df.to_csv(csv_path)

        # Create and save plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        plot_path = os.path.join(self.exp_dir, "plots", f"confusion_matrix_{task_type}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Store for later use
        self.confusion_matrices[task_type] = cm

        print(f"Confusion matrix saved: {plot_path}")

    def save_loss_curves(self):
        """Save loss curves as plots."""
        epochs = self.metrics_history['epoch']

        # Loss curves
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        if self.metrics_history['train_loss']:
            plt.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        if self.metrics_history['valid_loss']:
            plt.plot(epochs, self.metrics_history['valid_loss'], 'r-', label='Valid Loss')
        if self.metrics_history['test_loss']:
            plt.plot(epochs, self.metrics_history['test_loss'], 'g-', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)

        # Accuracy curves
        plt.subplot(1, 2, 2)
        if self.metrics_history['train_acc_emo']:
            plt.plot(epochs, self.metrics_history['train_acc_emo'], 'b-', label='Train Acc (Emo)')
        if self.metrics_history['valid_acc_emo']:
            plt.plot(epochs, self.metrics_history['valid_acc_emo'], 'r-', label='Valid Acc (Emo)')
        if self.metrics_history['test_acc_emo']:
            plt.plot(epochs, self.metrics_history['test_acc_emo'], 'g-', label='Test Acc (Emo)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, "plots", "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved: {plot_path}")

    def save_metrics_csv(self):
        """Save all metrics as CSV file."""
        df = pd.DataFrame(self.metrics_history)
        csv_path = os.path.join(self.exp_dir, "metrics", "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics CSV saved: {csv_path}")

    def save_classification_report(self, y_true, y_pred, labels, task_type="emotion"):
        """Save detailed classification report including accuracy."""
        # Generate the standard classification report as dict
        report_dict = classification_report(
            y_true, y_pred, target_names=labels, digits=4, output_dict=True, zero_division=0
        )
        report_text = classification_report(
            y_true, y_pred, target_names=labels, digits=4, zero_division=0
        )

        # Compute accuracy separately
        acc = accuracy_score(y_true, y_pred)

        # Save to CSV
        report_df = pd.DataFrame(report_dict).transpose()

        # Add accuracy as its own row with a single value column
        acc_row = pd.DataFrame({'accuracy': [acc]})
        report_df = pd.concat([report_df, acc_row.T.rename(columns={0: 'accuracy'})])

        csv_path = os.path.join(self.exp_dir, "metrics", f"classification_report_{task_type}.csv")
        report_df.to_csv(csv_path)

        # Save text report + accuracy
        txt_path = os.path.join(self.exp_dir, "metrics", f"classification_report_{task_type}.txt")
        with open(txt_path, 'w') as f:
            f.write(report_text)
            f.write(f"\n\nOverall Accuracy: {acc:.4f}\n")

        print(f"Classification report saved: {csv_path}")

    def save_experiment_config(self, args):
        """Save experiment configuration."""
        config = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'args': vars(args) if hasattr(args, '__dict__') else args
        }

        config_path = os.path.join(self.exp_dir, "experiment_config.pkl")
        with open(config_path, 'wb') as f:
            pk.dump(config, f)

        # Also save as readable text
        config_txt_path = os.path.join(self.exp_dir, "experiment_config.txt")
        with open(config_txt_path, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("Configuration:\n")
            for key, value in config['args'].items():
                f.write(f"  {key}: {value}\n")

        print(f"Experiment config saved: {config_path}")

    def finalize_experiment(self, best_labels_emo, best_preds_emo,
                            best_labels_sen=None, best_preds_sen=None,
                            emotion_labels=None, sentiment_labels=None):
        """Finalize experiment by saving all remaining artifacts."""

        # Default labels if not provided
        if emotion_labels is None:
            emotion_labels = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        if sentiment_labels is None:
            sentiment_labels = ['negative', 'neutral', 'positive']

        # Save confusion matrices and classification reports
        self.save_confusion_matrix(best_labels_emo, best_preds_emo,
                                   emotion_labels, "Emotion Recognition", "emotion")
        self.save_classification_report(best_labels_emo, best_preds_emo,
                                        emotion_labels, "emotion")

        if best_labels_sen is not None and best_preds_sen is not None:
            self.save_confusion_matrix(best_labels_sen, best_preds_sen,
                                       sentiment_labels, "Sentiment Analysis", "sentiment")
            self.save_classification_report(best_labels_sen, best_preds_sen,
                                            sentiment_labels, "sentiment")

        # Save training curves and metrics
        self.save_loss_curves()
        self.save_metrics_csv()

        # Save summary statistics
        self.save_summary_stats()

        print(f"Experiment finalized in: {self.exp_dir}")

    def save_summary_stats(self):
        """Save summary statistics of the experiment."""
        summary = {
            'best_test_acc_emo': max(self.metrics_history['test_acc_emo']) if self.metrics_history[
                'test_acc_emo'] else 0,
            'best_test_f1_emo': max(self.metrics_history['test_f1_emo']) if self.metrics_history['test_f1_emo'] else 0,
            'best_test_acc_sen': max(self.metrics_history['test_acc_sen']) if self.metrics_history[
                'test_acc_sen'] else 0,
            'best_test_f1_sen': max(self.metrics_history['test_f1_sen']) if self.metrics_history['test_f1_sen'] else 0,
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else 0,
            'final_test_loss': self.metrics_history['test_loss'][-1] if self.metrics_history['test_loss'] else 0,
            'total_epochs': len(self.metrics_history['epoch']),
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp
        }

        # Save as JSON-like format
        summary_path = os.path.join(self.exp_dir, "summary_stats.txt")
        with open(summary_path, 'w') as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"Summary statistics saved: {summary_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        return checkpoint


# Convenience function for easy integration
def create_model_saver(experiment_name, base_dir="experiments"):
    """Create a ModelSaver instance."""
    return ModelSaver(experiment_name, base_dir)