from collections import OrderedDict, deque
from functools import lru_cache, partial
import gc
import hashlib
import html
import math
import os
import pickle
import random
import re
from typing import Callable, Dict, List
import unicodedata

from datasets import load_dataset
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
)
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from simple import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    COMPRESSION_RATIO,
    DYNAMIC_K,
    EMBED_DIM,
    EXPERT_DIM,
    FF_HIDDEN_DIM,
    MAX_LENGTH,
    NUM_EPOCHS,
    NUM_EXPERTS,
    NUM_HEADS,
    NUM_LAYERS,
    TOP_K,
    WINDOW_SIZE,
)
from simple import (
    ActivationMonitor,
    BidirectionalEncoder,
    DeformableConv1d,
    EnhancedLSTM,
    LiquidEmbedding,
    EnhancedLocalAttentionWithGQA,
    ImprovedTransformerBlock,
    LiquidFoundationModelOptimized,
    MoELayer,
)

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuración global (simplificada)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Descargar recursos de NLTK si es necesario
nltk.download('wordnet', quiet=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Focal Loss para tareas de clasificación.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Calcula la Focal Loss.
        """
        # Calcula la entropía cruzada sin reducción
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # Probabilidad del modelo para la clase correcta

        # Calcula la Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Asegúrate de descargar los recursos necesarios de NLTK
nltk.download('punkt', quiet=True)

def verify_no_nans(batch):
    """
    Verifica que los tensores en el batch no contengan valores no finitos.
    Reemplaza NaNs e Infs por valores seguros o elimina el batch si es necesario.
    
    Args:
        batch (dict): Batch de datos a verificar
        
    Returns:
        dict: Batch limpio sin valores no finitos
    """
    # Flags para detectar si se encontraron NaNs o Infs
    has_nan = False
    has_inf = False
    nan_count = 0
    inf_count = 0
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            nan_in_key = torch.isnan(value).sum().item()
            inf_in_key = torch.isinf(value).sum().item()
            if nan_in_key > 0:
                print(f"NaNs encontrados en {key}: {nan_in_key} ocurrencias")
                has_nan = True
                nan_count += nan_in_key
            if inf_in_key > 0:
                print(f"Infs encontrados en {key}: {inf_in_key} ocurrencias")
                has_inf = True
                inf_count += inf_in_key
    
    # Acción a tomar si se encuentran NaNs o Infs
    if has_nan or has_inf:
        # Reemplazar NaNs e Infs por un valor seguro (por ejemplo, cero)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    batch[key] = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
                    print(f"Reemplazados {nan_count} NaNs e {inf_count} Infs en {key} por ceros.")
    return batch

def final_verification(batch):
    """
    Verifica nuevamente que no existan NaNs o Infs después de la limpieza.
    
    Args:
        batch (dict): Batch de datos a verificar
        
    Returns:
        dict: Batch verificado
        
    Raises:
        ValueError: Si se encuentran valores no finitos después de la limpieza
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError(f"Batch contiene valores no finitos en {key} después de la limpieza.")
    return batch

def prepare_data(max_samples=None, val_size=0.1, max_length=2048):
    """
    Prepara los datos del dataset TIGER-Lab/WebInstructSub con la estructura correcta.
    
    Args:
        max_samples (int, optional): Número máximo de muestras a usar
        val_size (float): Proporción del dataset para validación
        max_length (int): Longitud máxima de las secuencias
        
    Returns:
        tuple: (tokenizer, dataset dividido en train/val)
    """
    try:
        # Cargar dataset
        print("\n=== Iniciando Carga de Dataset ===")
        print("Cargando TIGER-Lab/WebInstructSub...")
        dataset = load_dataset("TIGER-Lab/WebInstructSub")
        print(f"Dataset cargado con éxito. Tamaño del conjunto de entrenamiento: {len(dataset['train'])}")
        
        # Mostrar estructura del dataset
        print("\nEstructura del dataset:")
        print("Columnas disponibles:", dataset['train'].column_names)
        sample = dataset['train'][0]
        print("\nEjemplo de datos:")
        for key, value in sample.items():
            print(f"{key}: {value[:100]}..." if isinstance(value, str) else f"{key}: {value}")
        
        # Configurar tokenizer
        print("\n=== Configuración del Tokenizer ===")
        print("Cargando tokenizer GPT2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {
            'pad_token': '[PAD]', 
            'eos_token': '<EOS>', 
            'bos_token': '<BOS>',
            'sep_token': '[SEP]'
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens)
        print(f"Tokens especiales agregados: {num_added_toks}")
        print(f"Vocabulario actual: {len(tokenizer)} tokens")
        print("Tokens especiales:")
        for token_name, token in special_tokens.items():
            print(f"  - {token_name}: {token} (ID: {tokenizer.convert_tokens_to_ids(token)})")

        # Limitar muestras
        if max_samples is not None and max_samples < len(dataset['train']):
            dataset['train'] = dataset['train'].select(range(max_samples))
            print(f"\nDataset limitado a {max_samples} muestras")
            print(f"Nuevo tamaño del conjunto de entrenamiento: {len(dataset['train'])}")

        def improved_preprocess_text(examples):
            """
            Preprocesa el texto aplicando varias transformaciones de limpieza.
            """
            def clean_text(text):
                if not isinstance(text, str):
                    print(f"Advertencia: texto no válido encontrado: {type(text)}")
                    return ""
                
                try:
                    # Decodificar entidades HTML
                    text = html.unescape(text)
                    
                    # Normalizar caracteres Unicode
                    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
                    
                    # Eliminar URLs
                    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                    
                    # Normalizar espacios en blanco
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Eliminar espacios antes de la puntuación
                    text = re.sub(r'\s([?.!,:](?:\s|$))', r'\1', text)
                    
                    return text
                except Exception as e:
                    print(f"Error en clean_text: {str(e)}")
                    return ""

            print("\n=== Iniciando Preprocesamiento de Texto ===")
            cleaned_questions = []
            cleaned_answers = []
            
            for question, answer in zip(examples['question'], examples['answer']):
                clean_q = clean_text(question)
                clean_a = clean_text(answer)
                
                if not clean_q.strip() or not clean_a.strip():
                    print("Advertencia: pregunta o respuesta vacía después de limpieza")
                    clean_q = question if not clean_q.strip() else clean_q
                    clean_a = answer if not clean_a.strip() else clean_a
                
                cleaned_questions.append(clean_q)
                cleaned_answers.append(clean_a)
            
            return {
                'cleaned_question': cleaned_questions,
                'cleaned_answer': cleaned_answers
            }

        def sample_text(examples, max_length):
            """
            Muestrea texto con logging detallado.
            """
            print("\n=== Iniciando Muestreo de Texto ===")
            sampled_questions = []
            sampled_answers = []
            
            for q, a in zip(examples['cleaned_question'], examples['cleaned_answer']):
                total_length = len(q) + len(a) + 3  # BOS, SEP, EOS
                
                if total_length > max_length:
                    # Distribuir el espacio proporcionalmente
                    q_ratio = len(q) / total_length
                    a_ratio = len(a) / total_length
                    
                    max_q_len = int((max_length - 3) * q_ratio)
                    max_a_len = int((max_length - 3) * a_ratio)
                    
                    q = q[:max_q_len]
                    a = a[:max_a_len]
                
                sampled_questions.append(q)
                sampled_answers.append(a)
            
            return {
                'sampled_question': sampled_questions,
                'sampled_answer': sampled_answers
            }

        def tokenize_data(examples, tokenizer, max_length=2048, verbose=True):
            """
            Tokeniza los textos y prepara las entradas para el modelo, mostrando ejemplos
            de tokenización para cada batch.
            
            Args:
                examples: Batch de ejemplos a tokenizar
                tokenizer: Tokenizador a utilizar
                max_length: Longitud máxima de secuencia
                verbose: Si es True, muestra ejemplos de tokenización
            """
            try:
                # Combinar pregunta y respuesta con tokens especiales
                combined_texts = [
                    f"{tokenizer.bos_token}{q}{tokenizer.sep_token}{a}{tokenizer.eos_token}"
                    for q, a in zip(examples['sampled_question'], examples['sampled_answer'])
                ]
                
                # Tokenización
                tokens = tokenizer(
                    combined_texts,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_attention_mask=True
                )
                
                # Preparar decoder_input_ids y labels
                decoder_input_ids = []
                labels = []
                
                for ids in tokens['input_ids']:
                    sep_pos = ids.index(tokenizer.sep_token_id)
                    
                    # decoder_input_ids: desde SEP hasta EOS-1
                    decoder_ids = [tokenizer.bos_token_id] + ids[sep_pos+1:-1]
                    # Padding
                    decoder_ids = decoder_ids + [tokenizer.pad_token_id] * (max_length - 1 - len(decoder_ids))
                    
                    # labels: desde SEP+1 hasta EOS
                    label_ids = ids[sep_pos+1:]
                    # Padding con -100
                    label_ids = label_ids + [-100] * (max_length - 1 - len(label_ids))
                    
                    decoder_input_ids.append(decoder_ids)
                    labels.append(label_ids)
                
                tokens['decoder_input_ids'] = decoder_input_ids
                tokens['labels'] = labels
                
                # Mostrar ejemplo de tokenización para el primer elemento del batch
                if verbose and len(tokens['input_ids']) > 0:
                    print("\n=== Ejemplo de Tokenización del Batch ===")
                    print("\n1. Texto Original:")
                    print(f"Pregunta: {examples['sampled_question'][0]}")
                    print(f"Respuesta: {examples['sampled_answer'][0]}")
                    
                    print("\n2. Texto Combinado:")
                    print(combined_texts[0])
                    
                    print("\n3. Tokens (IDs):")
                    print(f"Input IDs: {tokens['input_ids'][0][:50]}...")
                    print(f"Decoder Input IDs: {tokens['decoder_input_ids'][0][:50]}...")
                    print(f"Labels: {[l for l in tokens['labels'][0][:50] if l != -100]}...")
                    
                    print("\n4. Tokens Decodificados:")
                    print(f"Input: {tokenizer.decode(tokens['input_ids'][0])[:100]}...")
                    print(f"Decoder Input: {tokenizer.decode(tokens['decoder_input_ids'][0])[:100]}...")
                    print(f"Labels (sin padding): {tokenizer.decode([l for l in tokens['labels'][0] if l != -100])[:100]}...")
                    
                    print("\n5. Estadísticas:")
                    print(f"Longitud total de input_ids: {len(tokens['input_ids'][0])}")
                    print(f"Longitud total de decoder_input_ids: {len(tokens['decoder_input_ids'][0])}")
                    print(f"Tokens válidos en labels: {sum(1 for l in tokens['labels'][0] if l != -100)}")
                    print(f"Tokens de padding en labels: {sum(1 for l in tokens['labels'][0] if l == -100)}")
                    
                    print("\n6. Tokens Especiales:")
                    special_tokens = {
                        'BOS': tokenizer.bos_token_id,
                        'EOS': tokenizer.eos_token_id,
                        'SEP': tokenizer.sep_token_id,
                        'PAD': tokenizer.pad_token_id
                    }
                    for name, token_id in special_tokens.items():
                        count = tokens['input_ids'][0].count(token_id)
                        print(f"{name} token (ID: {token_id}) aparece {count} veces")
                    
                    print("\n=== Fin del Ejemplo de Tokenización ===\n")
                
                return tokens
                    
            except Exception as e:
                print(f"Error en tokenize_data: {str(e)}")
                return {
                    'input_ids': [[tokenizer.pad_token_id] * max_length],
                    'decoder_input_ids': [[tokenizer.pad_token_id] * (max_length - 1)],
                    'labels': [[-100] * (max_length - 1)],
                    'attention_mask': [[0] * max_length]
                }

        # Aplicar pipeline de procesamiento
        print("\n=== Iniciando Pipeline de Procesamiento ===")
        
        print("\n1. Preprocesamiento...")
        processed_dataset = dataset['train'].map(
            improved_preprocess_text,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Preprocesando textos"
        )
        
        print("\n2. Muestreo...")
        sampled_dataset = processed_dataset.map(
            lambda x: sample_text(x, max_length),
            batched=True,
            remove_columns=processed_dataset.column_names,
            desc="Muestreando textos"
        )
        
        print("\n3. Tokenización...")
        tokenized_dataset = sampled_dataset.map(
            lambda x: tokenize_data(x, tokenizer, max_length=max_length),
            batched=True,
            remove_columns=sampled_dataset.column_names,
            desc="Tokenizando textos"
        )
        
        # Verificaciones y formato PyTorch
        print("\n=== Verificaciones Finales ===")
        tokenized_dataset = tokenized_dataset.map(
            verify_no_nans,
            batched=True,
            batch_size=1000,
            desc="Verificando valores no finitos"
        )
        
        tokenized_dataset = tokenized_dataset.map(
            final_verification,
            batched=True,
            batch_size=1000,
            desc="Verificación final"
        )
        
        print("\n=== Configurando Formato PyTorch ===")
        tokenized_dataset.set_format(
            type='torch',
            columns=['input_ids', 'decoder_input_ids', 'labels', 'attention_mask']
        )
        
        # División del dataset
        print("\n=== Dividiendo Dataset ===")
        train_val_dataset = tokenized_dataset.train_test_split(test_size=val_size)
        
        print("\nTamaños finales:")
        print(f"Training: {len(train_val_dataset['train'])}")
        print(f"Validation: {len(train_val_dataset['test'])}")
        
        return tokenizer, train_val_dataset
        
    except Exception as e:
        print(f"Error en prepare_data: {str(e)}")
        raise e        
def train_one_batch(batch, model, ce_criterion, focal_criterion, optimizer, scaler, device, accumulation_steps, metrics_tracker):
    """
    Entrena el modelo con un solo batch y actualiza las métricas, incluyendo el vocabulario.
    """
    try:
        encoder_input_ids = batch['input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():
            outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(
                encoder_input_ids, decoder_input_ids
            )

            logits = outputs.reshape(-1, outputs.size(-1))
            labels_flat = labels.reshape(-1)
            mask = labels_flat != -100
            valid_logits = logits[mask]
            valid_labels = labels_flat[mask]

            if valid_labels.numel() > 0:
                # Calcular pérdidas
                ce_loss = ce_criterion(valid_logits, valid_labels)
                focal_loss = focal_criterion(valid_logits, valid_labels)
                total_loss = (
                    ce_loss +
                    focal_loss +
                    recon_loss_enc +
                    recon_loss_dec +
                    entropy_loss_enc +
                    entropy_loss_dec 
                )

                # Escalar pérdida y backward
                scaled_loss = total_loss / accumulation_steps
                scaler.scale(scaled_loss).backward()

                # Actualizar métricas de entrenamiento
                predictions = valid_logits.argmax(dim=-1)
                metrics_tracker['total_loss'] += total_loss.item() * valid_labels.numel()
                metrics_tracker['ce_loss'] += ce_loss.item() * valid_labels.numel()
                metrics_tracker['focal_loss'] += focal_loss.item() * valid_labels.numel()
                metrics_tracker['recon_loss_enc'] += recon_loss_enc.item() * valid_labels.numel()
                metrics_tracker['recon_loss_dec'] += recon_loss_dec.item() * valid_labels.numel()
                metrics_tracker['entropy_loss_enc'] += entropy_loss_enc.item() * valid_labels.numel()
                metrics_tracker['entropy_loss_dec'] += entropy_loss_dec.item() * valid_labels.numel()
                metrics_tracker['valid_tokens'] += valid_labels.numel()

                # Accuracy metrics
                correct_tokens = (predictions == valid_labels).sum().item()
                metrics_tracker['correct_tokens'] += correct_tokens

                batch_predictions = outputs.argmax(dim=-1)
                correct_sequences = (batch_predictions == labels).all(dim=1).sum().item()
                metrics_tracker['correct_sequences'] += correct_sequences
                metrics_tracker['total_sequences'] += labels.size(0)

                # Top-k accuracy
                for k in [1, 3, 5, 10]:
                    top_k_preds = torch.topk(valid_logits, k, dim=-1).indices
                    correct_k = sum(valid_labels[i] in top_k_preds[i] for i in range(len(valid_labels)))
                    metrics_tracker[f'top_k_accuracy_{k}'] += float(correct_k)

                # Diversidad y vocabulario
                pred_tokens = predictions.cpu().tolist()
                true_tokens = valid_labels.cpu().tolist()
                metrics_tracker['pred_tokens'].extend(pred_tokens)
                metrics_tracker['true_tokens'].extend(true_tokens)
                metrics_tracker['unique_unigrams'].update(pred_tokens)
                
                # Actualizar vocabulario con tokens predichos y verdaderos
                if isinstance(metrics_tracker['vocab_tokens'], set):
                    metrics_tracker['vocab_tokens'].update(pred_tokens)
                    metrics_tracker['vocab_tokens'].update(true_tokens)

                # Actualizar bigramas
                for i in range(len(pred_tokens) - 1):
                    metrics_tracker['unique_bigrams'].add((pred_tokens[i], pred_tokens[i + 1]))

                # Fluidez
                seq_lengths = (labels != -100).sum(dim=1).cpu().tolist()
                metrics_tracker['sequence_lengths'].extend(seq_lengths)

                # Perplejidad
                log_probs = F.log_softmax(valid_logits, dim=-1)
                token_perplexities = torch.exp(-torch.gather(
                    log_probs, 1, valid_labels.unsqueeze(1)
                )).squeeze(1)
                metrics_tracker['perplexities'].extend(token_perplexities.cpu().tolist())

    except Exception as e:
        print(f"Error en train_one_batch: {str(e)}")
        raise e

def evaluate(model, data_loader, ce_criterion, focal_criterion, device):
    """
    Evalúa el modelo en un conjunto de datos.
    
    Args:
        model: Modelo a evaluar.
        data_loader: DataLoader del conjunto de datos.
        ce_criterion: Función de pérdida CrossEntropyLoss.
        focal_criterion: Función de pérdida FocalLoss.
        device: Dispositivo de evaluación.
    
    Returns:
        dict: Métricas calculadas.
    """
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'ce_loss': 0.0,
        'focal_loss': 0.0,
        'recon_loss_enc': 0.0,
        'recon_loss_dec': 0.0,
        'entropy_loss_enc': 0.0,
        'entropy_loss_dec': 0.0,
        'valid_tokens': 0,
        'correct_tokens': 0,
        'correct_sequences': 0,
        'total_sequences': 0,
        'pred_tokens': [],
        'true_tokens': [],
        'unique_unigrams': set(),
        'unique_bigrams': set(),
        'sequence_lengths': [],
        'perplexities': [],
        'top_k_accuracy_1': 0.0,
        'top_k_accuracy_3': 0.0,
        'top_k_accuracy_5': 0.0,
        'top_k_accuracy_10': 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluando"):
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():
                outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(
                    encoder_input_ids, decoder_input_ids
                )

                logits = outputs.reshape(-1, outputs.size(-1))
                labels_flat = labels.reshape(-1)
                mask = labels_flat != -100
                valid_logits = logits[mask]
                valid_labels = labels_flat[mask]

                if valid_labels.numel() > 0:
                    # Calcular pérdidas
                    ce_loss = ce_criterion(valid_logits, valid_labels)
                    focal_loss = focal_criterion(valid_logits, valid_labels)
                    # Calcular la pérdida total con ponderación
                    total_loss = (
                        ce_loss +
                        focal_loss +
                        recon_loss_enc +
                        recon_loss_dec +
                        entropy_loss_enc +
                        entropy_loss_dec 
                    )

                    # Actualizar métricas
                    metrics['total_loss'] += total_loss.item() * valid_labels.numel()
                    metrics['ce_loss'] += ce_loss.item() * valid_labels.numel()
                    metrics['focal_loss'] += focal_loss.item() * valid_labels.numel()
                    metrics['recon_loss_enc'] += recon_loss_enc.item() * valid_labels.numel()
                    metrics['recon_loss_dec'] += recon_loss_dec.item() * valid_labels.numel()
                    metrics['entropy_loss_enc'] += entropy_loss_enc.item() * valid_labels.numel()
                    metrics['entropy_loss_dec'] += entropy_loss_dec.item() * valid_labels.numel()
                    metrics['valid_tokens'] += valid_labels.numel()

                    # Accuracy metrics
                    predictions = valid_logits.argmax(dim=-1)
                    metrics['correct_tokens'] += (predictions == valid_labels).sum().item()

                    batch_predictions = outputs.argmax(dim=-1)
                    metrics['correct_sequences'] += (batch_predictions == labels).all(dim=1).sum().item()
                    metrics['total_sequences'] += labels.size(0)

                    # Top-k accuracy
                    for k in [1, 3, 5, 10]:
                        top_k_preds = torch.topk(valid_logits, k, dim=-1).indices
                        correct_k = sum(valid_labels[i] in top_k_preds[i] for i in range(len(valid_labels)))
                        metrics[f'top_k_accuracy_{k}'] += float(correct_k)

                    # Diversidad
                    pred_tokens = predictions.cpu().tolist()
                    true_tokens = valid_labels.cpu().tolist()
                    metrics['pred_tokens'].extend(pred_tokens)
                    metrics['true_tokens'].extend(true_tokens)
                    metrics['unique_unigrams'].update(pred_tokens)
                    
                    # Actualizar bigramas
                    for i in range(len(pred_tokens) - 1):
                        metrics['unique_bigrams'].add((pred_tokens[i], pred_tokens[i + 1]))

                    # Fluidez
                    seq_lengths = (labels != -100).sum(dim=1).cpu().tolist()
                    metrics['sequence_lengths'].extend(seq_lengths)

                    # Perplejidad
                    log_probs = F.log_softmax(valid_logits, dim=-1)
                    token_perplexities = torch.exp(-torch.gather(
                        log_probs, 1, valid_labels.unsqueeze(1)
                    )).squeeze(1)
                    metrics['perplexities'].extend(token_perplexities.cpu().tolist())

    # Calcular métricas finales si hay tokens válidos
    if metrics['valid_tokens'] > 0:
        final_metrics = {
            'total_loss': metrics['total_loss'] / metrics['valid_tokens'],
            'ce_loss': metrics['ce_loss'] / metrics['valid_tokens'],
            'focal_loss': metrics['focal_loss'] / metrics['valid_tokens'],
            'recon_loss_enc': metrics['recon_loss_enc'] / metrics['valid_tokens'],
            'recon_loss_dec': metrics['recon_loss_dec'] / metrics['valid_tokens'],
            'entropy_loss_enc': metrics['entropy_loss_enc'] / metrics['valid_tokens'],
            'entropy_loss_dec': metrics['entropy_loss_dec'] / metrics['valid_tokens'],
            'token_accuracy': metrics['correct_tokens'] / metrics['valid_tokens'],
            'sequence_accuracy': metrics['correct_sequences'] / metrics['total_sequences'] if metrics['total_sequences'] > 0 else 0.0,
            'top_k_accuracy_1': metrics['top_k_accuracy_1'] / metrics['valid_tokens'],
            'top_k_accuracy_3': metrics['top_k_accuracy_3'] / metrics['valid_tokens'],
            'top_k_accuracy_5': metrics['top_k_accuracy_5'] / metrics['valid_tokens'],
            'top_k_accuracy_10': metrics['top_k_accuracy_10'] / metrics['valid_tokens'],
            'distinct_1': len(metrics['unique_unigrams']) / len(metrics['pred_tokens']) if metrics['pred_tokens'] else 0.0,
            'distinct_2': len(metrics['unique_bigrams']) / max(1, len(metrics['pred_tokens']) - 1) if metrics['pred_tokens'] else 0.0,
            'avg_sequence_length': float(np.mean(metrics['sequence_lengths'])) if metrics['sequence_lengths'] else 0.0,
            'perplexity': float(np.mean(metrics['perplexities'])) if metrics['perplexities'] else 0.0,
            'vocab_size': len(metrics['unique_unigrams']),
            'pred_tokens': metrics['pred_tokens'],      # Añadido
            'true_tokens': metrics['true_tokens'],      # Añadido
        }

        # Calcular BLEU scores
        if metrics['pred_tokens'] and metrics['true_tokens']:
            try:
                smoothing = SmoothingFunction().method1
                pred_tokens_str = [str(token) for token in metrics['pred_tokens']]
                true_tokens_str = [str(token) for token in metrics['true_tokens']]

                for n in range(1, 5):
                    weights = tuple([1.0/n] * n + [0.0] * (4-n))
                    bleu_scores = [
                        sentence_bleu([ref], hyp, weights=weights, smoothing_function=smoothing)
                        for ref, hyp in zip([true_tokens_str], [pred_tokens_str])
                    ]
                    final_metrics[f'bleu_{n}'] = np.mean(bleu_scores)
            except Exception as e:
                print(f"Error calculando BLEU: {str(e)}")
                for n in range(1, 5):
                    final_metrics[f'bleu_{n}'] = 0.0
        else:
            for n in range(1, 5):
                final_metrics[f'bleu_{n}'] = 0.0

    else:
        # Establecer valores por defecto cuando no hay tokens válidos
        final_metrics = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'focal_loss': 0.0,
            'recon_loss_enc': 0.0,
            'recon_loss_dec': 0.0,
            'entropy_loss_enc': 0.0,
            'entropy_loss_dec': 0.0,
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'top_k_accuracy_1': 0.0,
            'top_k_accuracy_3': 0.0,
            'top_k_accuracy_5': 0.0,
            'top_k_accuracy_10': 0.0,
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'avg_sequence_length': 0.0,
            'perplexity': 0.0,
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'vocab_size': 0,
            'pred_tokens': [],      # Añadido
            'true_tokens': [],      # Añadido
        }

    return final_metrics


def log_metrics(writer, train_metrics, val_metrics, epoch):
    """
    Registra las métricas en TensorBoard y en la consola.
    
    Args:
        writer: SummaryWriter de TensorBoard.
        train_metrics (dict): Métricas de entrenamiento.
        val_metrics (dict): Métricas de validación.
        epoch (int): Número de la época actual.
    
    Returns:
        None
    """
    # Logging a TensorBoard
    # Métricas básicas
    writer.add_scalar('Loss/train', train_metrics['total_loss'], epoch)
    writer.add_scalar('Loss/val', val_metrics['total_loss'], epoch)
    writer.add_scalar('Perplexity/val', val_metrics['perplexity'], epoch)

    # Accuracy
    writer.add_scalar('Accuracy/token', train_metrics['token_accuracy'], epoch)
    writer.add_scalar('Accuracy/sequence', train_metrics['sequence_accuracy'], epoch)
    for k in [1, 3, 5, 10]:
        writer.add_scalar(f'Accuracy/top_{k}', train_metrics[f'top_k_accuracy_{k}'], epoch)

    # Diversidad
    writer.add_scalar('Diversity/distinct_1', train_metrics['distinct_1'], epoch)
    writer.add_scalar('Diversity/distinct_2', train_metrics['distinct_2'], epoch)

    # Fluidez
    writer.add_scalar('Fluency/avg_sequence_length', train_metrics['avg_sequence_length'], epoch)

    # BLEU
    for n in range(1, 5):
        writer.add_scalar(f'BLEU/bleu_{n}', train_metrics[f'bleu_{n}'], epoch)

    # Vocabulario
    writer.add_scalar('Vocabulary/vocab_size', train_metrics['vocab_size'], epoch)

    # Logging en consola
    print("\n--- Métricas de Entrenamiento ---")
    print(f"Total Loss: {train_metrics['total_loss']:.4f}")
    print(f"  - CrossEntropy Loss: {train_metrics['ce_loss']:.4f}")
    print(f"  - Focal Loss: {train_metrics['focal_loss']:.4f}")
    print(f"  - Recon Loss Enc: {train_metrics['recon_loss_enc']:.4f}")
    print(f"  - Recon Loss Dec: {train_metrics['recon_loss_dec']:.4f}")
    print(f"  - Entropy Loss Enc: {train_metrics['entropy_loss_enc']:.4f}")
    print(f"  - Entropy Loss Dec: {train_metrics['entropy_loss_dec']:.4f}")

    print("\n--- Métricas de Validación ---")
    print(f"Total Loss: {val_metrics['total_loss']:.4f}")
    print(f"  - CrossEntropy Loss: {val_metrics['ce_loss']:.4f}")
    print(f"  - Focal Loss: {val_metrics['focal_loss']:.4f}")
    print(f"  - Recon Loss Enc: {val_metrics['recon_loss_enc']:.4f}")
    print(f"  - Recon Loss Dec: {val_metrics['recon_loss_dec']:.4f}")
    print(f"  - Entropy Loss Enc: {val_metrics['entropy_loss_enc']:.4f}")
    print(f"  - Entropy Loss Dec: {val_metrics['entropy_loss_dec']:.4f}")

    print("\n--- Métricas de Accuracy ---")
    print(f"Token Accuracy: {train_metrics['token_accuracy']:.4f}")
    print(f"Sequence Accuracy: {train_metrics['sequence_accuracy']:.4f}")
    for k in [1, 3, 5, 10]:
        print(f"Top-{k} Accuracy: {train_metrics[f'top_k_accuracy_{k}']:.4f}")

    print("\n--- Métricas de Diversidad ---")
    print(f"Distinct-1: {train_metrics['distinct_1']:.4f}")
    print(f"Distinct-2: {train_metrics['distinct_2']:.4f}")

    print("\n--- Métricas de Fluidez ---")
    print(f"Avg Sequence Length: {train_metrics['avg_sequence_length']:.2f}")

    print("\n--- BLEU Scores ---")
    for n in range(1, 5):
        print(f"BLEU-{n}: {train_metrics[f'bleu_{n}']:.4f}")

    print("\n--- Métricas de Vocabulario ---")
    print(f"Vocab Size: {train_metrics['vocab_size']}")
def train_model(
    model, 
    train_loader, 
    val_loader, 
    ce_criterion, 
    focal_criterion, 
    optimizer, 
    scheduler, 
    scaler, 
    device, 
    num_epochs, 
    accumulation_steps=10, 
    monitor=None, 
    debug_loss_components=True
):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    epoch = 0  # Inicialización de epoch

    # Listas para almacenar el historial de métricas
    train_losses = []
    val_losses = []
    val_perplexities = []

    # Inicializar el tracker de métricas con todos los campos necesarios
    metrics_tracker = {
        'total_loss': 0.0,
        'ce_loss': 0.0,
        'focal_loss': 0.0,
        'recon_loss_enc': 0.0,
        'recon_loss_dec': 0.0,
        'entropy_loss_enc': 0.0,
        'entropy_loss_dec': 0.0,
        'valid_tokens': 0,
        'correct_tokens': 0,
        'correct_sequences': 0,
        'total_sequences': 0,
        'pred_tokens': [],
        'true_tokens': [],
        'unique_unigrams': set(),
        'unique_bigrams': set(),
        'vocab_tokens': set(),  # Conjunto para tracking del vocabulario
        'sequence_lengths': [],
        'perplexities': [],
        'top_k_accuracy_1': 0.0,
        'top_k_accuracy_3': 0.0,
        'top_k_accuracy_5': 0.0,
        'top_k_accuracy_10': 0.0,
    }

    def reset_metrics_tracker():
        """Reinicia todas las métricas a sus valores iniciales."""
        nonlocal metrics_tracker
        metrics_tracker = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'focal_loss': 0.0,
            'recon_loss_enc': 0.0,
            'recon_loss_dec': 0.0,
            'entropy_loss_enc': 0.0,
            'entropy_loss_dec': 0.0,
            'valid_tokens': 0,
            'correct_tokens': 0,
            'correct_sequences': 0,
            'total_sequences': 0,
            'pred_tokens': [],
            'true_tokens': [],
            'unique_unigrams': set(),
            'unique_bigrams': set(),
            'vocab_tokens': set(),  # Reiniciar el conjunto de vocabulario
            'sequence_lengths': [],
            'perplexities': [],
            'top_k_accuracy_1': 0.0,
            'top_k_accuracy_3': 0.0,
            'top_k_accuracy_5': 0.0,
            'top_k_accuracy_10': 0.0,
        }

    def calculate_metrics(model, data_loader, is_training=True):
        """
        Calcula todas las métricas sobre un conjunto de datos.

        Args:
            model: El modelo a evaluar.
            data_loader: DataLoader del conjunto de datos.
            is_training: Booleano que indica si es durante el entrenamiento.

        Returns:
            dict: Métricas calculadas.
        """
        return evaluate(model, data_loader, ce_criterion, focal_criterion, device)

    try:
        # Verificar que todos los parámetros del modelo tienen requires_grad=True
        print("Verificando parámetros del modelo que no requieren gradientes...")
        frozen_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_params.append(name)
        if frozen_params:
            print("¡Advertencia! Los siguientes parámetros no requieren gradientes y no se actualizarán durante el entrenamiento:")
            for name in frozen_params:
                print(f" - {name}")
            print("Por favor, revisa la configuración de tu modelo para asegurarte de que todos los parámetros que deseas entrenar tienen 'requires_grad=True'.")
            raise ValueError("Algunos parámetros del modelo no requieren gradientes. Revisa las advertencias arriba.")
        else:
            print("Todos los parámetros del modelo requieren gradientes. Continuando con el entrenamiento.")

        while epoch < num_epochs:
            model.train()  # Asegurar modo de entrenamiento al inicio de cada epoch
            print(f"\n--- Epoch {epoch + 1} ---")
            reset_metrics_tracker()

            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in loop:
                try:
                    train_one_batch(batch, model, ce_criterion, focal_criterion, optimizer, scaler, device, accumulation_steps, metrics_tracker)

                    # Actualización de pesos
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()  # Si usas CosineAnnealingWarmRestarts
                        optimizer.zero_grad()

                    # Actualizar barra de progreso
                    if metrics_tracker['valid_tokens'] > 0:
                        current_loss = metrics_tracker['total_loss'] / metrics_tracker['valid_tokens']
                        loop.set_postfix(loss=f"{current_loss:.4f}")

                except Exception as e:
                    print(f"Error en batch {batch_idx}: {str(e)}")
                    continue

            # Fin de epoch - Evaluación
            try:
                # Liberar memoria del entrenamiento acumulado
                torch.cuda.empty_cache()
                gc.collect()

                # Calcular métricas de entrenamiento y validación
                train_metrics = calculate_metrics(model, train_loader, is_training=True)
                val_metrics = calculate_metrics(model, val_loader, is_training=False)

                # Almacenar métricas básicas
                train_losses.append(train_metrics['total_loss'])
                val_losses.append(val_metrics['total_loss'])
                val_perplexities.append(val_metrics['perplexity'])

                # Logging detallado
                print(f"\nEpoch {epoch + 1} completada. Evaluando modelo...")

                if debug_loss_components:
                    print("\n--- Métricas de Entrenamiento ---")
                    print(f"Total Loss: {train_metrics['total_loss']:.4f}")
                    print(f"  - CrossEntropy Loss: {train_metrics['ce_loss']:.4f}")
                    print(f"  - Focal Loss: {train_metrics['focal_loss']:.4f}")
                    print(f"  - Recon Loss Enc: {train_metrics['recon_loss_enc']:.4f}")
                    print(f"  - Recon Loss Dec: {train_metrics['recon_loss_dec']:.4f}")
                    print(f"  - Entropy Loss Enc: {train_metrics['entropy_loss_enc']:.4f}")
                    print(f"  - Entropy Loss Dec: {train_metrics['entropy_loss_dec']:.4f}")

                    print("\n--- Métricas de Validación ---")
                    print(f"Total Loss: {val_metrics['total_loss']:.4f}")
                    print(f"  - CrossEntropy Loss: {val_metrics['ce_loss']:.4f}")
                    print(f"  - Focal Loss: {val_metrics['focal_loss']:.4f}")
                    print(f"  - Recon Loss Enc: {val_metrics['recon_loss_enc']:.4f}")
                    print(f"  - Recon Loss Dec: {val_metrics['recon_loss_dec']:.4f}")
                    print(f"  - Entropy Loss Enc: {val_metrics['entropy_loss_enc']:.4f}")
                    print(f"  - Entropy Loss Dec: {val_metrics['entropy_loss_dec']:.4f}")

                    print("\n--- Métricas de Accuracy ---")
                    print(f"Token Accuracy: {train_metrics['token_accuracy']:.4f}")
                    print(f"Sequence Accuracy: {train_metrics['sequence_accuracy']:.4f}")
                    for k in [1, 3, 5, 10]:
                        print(f"Top-{k} Accuracy: {train_metrics[f'top_k_accuracy_{k}']:.4f}")

                    print("\n--- Métricas de Diversidad ---")
                    print(f"Distinct-1: {train_metrics['distinct_1']:.4f}")
                    print(f"Distinct-2: {train_metrics['distinct_2']:.4f}")

                    print("\n--- Métricas de Fluidez ---")
                    print(f"Avg Sequence Length: {train_metrics['avg_sequence_length']:.2f}")

                    print("\n--- BLEU Scores ---")
                    for n in range(1, 5):
                        print(f"BLEU-{n}: {train_metrics[f'bleu_{n}']:.4f}")

                    print("\n--- Métricas de Vocabulario ---")
                    print(f"Vocab Size: {train_metrics['vocab_size']}")

                    # **Añadir Inspección de Predicciones y Etiquetas**
                    print("\n--- Ejemplo de Predicciones vs Etiquetas ---")
                    if 'pred_tokens' in train_metrics and 'true_tokens' in train_metrics:
                        example_pred = train_metrics['pred_tokens'][:10]
                        example_true = train_metrics['true_tokens'][:10]
                        print(f"Predicciones: {example_pred}")
                        print(f"Etiquetas: {example_true}")
                    else:
                        print("Advertencia: 'pred_tokens' o 'true_tokens' no están presentes en las métricas de entrenamiento.")

                # Logging a TensorBoard y consola
                log_metrics(writer, train_metrics, val_metrics, epoch + 1)

                # Early stopping y guardado del mejor modelo
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    no_improve = 0
                    # Guardar mejor modelo con métricas extendidas
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'metrics': {
                            'train': train_metrics,
                            'val': val_metrics
                        },
                        'train_history': {
                            'losses': train_losses
                        },
                        'val_history': {
                            'losses': val_losses,
                            'perplexities': val_perplexities
                        }
                    }, 'best_model.pth')
                    print("\nGuardado nuevo mejor modelo!")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("\nEarly stopping triggered")
                        break

                # Limpiar variables de evaluación para liberar memoria
                del train_metrics, val_metrics
                torch.cuda.empty_cache()
                gc.collect()

                # Fin de época
                epoch += 1
                print(f"\nEpoch {epoch} completada. Batches procesados: {len(train_loader)}")

            except KeyboardInterrupt:
                print("\nEntrenamiento interrumpido por el usuario")
                break
    finally:
        writer.close()
        if monitor:
            monitor.remove_hooks()

        # Guardar visualizaciones finales
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 5))

            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Val')
            plt.title('Evolución de la Pérdida')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()

            # Perplexity
            plt.subplot(1, 2, 2)
            plt.plot(val_perplexities)
            plt.title('Evolución de la Perplejidad')
            plt.xlabel('Época')
            plt.ylabel('Perplejidad')
            plt.yscale('log')

            plt.tight_layout()
            plt.savefig('training_evolution.png')
            plt.close()

            print("\nVisualizaciones finales guardadas en 'training_evolution.png'")
        except Exception as e:
            print(f"Error generando visualizaciones finales: {str(e)}")

def main_analysis():
    print("\n=== Iniciando Análisis Principal ===")
    
    # Configuración inicial
    config = {
        'max_samples': 25,
        'val_size': 0.1,
        'max_length': 150,
        'batch_size': BATCH_SIZE,
        'num_epochs': 5,  # Ajusta según tus necesidades
        'lr': 1e-4,
        'temperature': 0.7,
        'top_k': 50,
        'min_length': 10
    }
    
    print("\nConfiguración:")
    for key, value in config.items():
        print(f"- {key}: {value}")

    # Preparación de datos y modelo
    tokenizer, datasets = prepare_data(max_samples=config['max_samples'], 
                                        val_size=config['val_size'], 
                                        max_length=config['max_length'])
    VOCAB_SIZE = len(tokenizer)
    print(f"\nTamaño del vocabulario: {VOCAB_SIZE}")

    train_loader = DataLoader(datasets['train'], batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets['test'], batch_size=config['batch_size'], shuffle=False)

    model = LiquidFoundationModelOptimized(vocab_size=VOCAB_SIZE).to(device)
    print(f"\nModelo creado en dispositivo: {device}")

    # Criterios de pérdida
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    focal_criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100)
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = GradScaler()
    monitor = ActivationMonitor(model)

    # Configuración de entrenamiento
    print("\nEstadísticas de entrenamiento:")
    print(f"- Batches por época: {len(train_loader)}")
    print(f"- Total épocas objetivo: {config['num_epochs']}")
    print(f"- Intervalo de evaluación: al final de cada época")

    # Entrenamiento
    print("\n=== Iniciando Entrenamiento ===")
    train_losses, val_losses, val_perplexities = train_model(
        model, train_loader, val_loader, ce_criterion, focal_criterion, 
        optimizer, scheduler, scaler, device, 
        num_epochs=config['num_epochs'],
        accumulation_steps=ACCUMULATION_STEPS, 
        monitor=monitor
    )

    # Análisis post-entrenamiento
    print("\n=== Análisis Post-Entrenamiento ===")
    prompt = "Ejemplo de prompt para análisis de tokens."
    train_stats, val_stats = analyze_token_transformations_extended(
        model, tokenizer, prompt, train_loader, val_loader,
        max_length=config['max_length'],
        min_length=config['min_length'],
        temperature=config['temperature'],
        top_k=config['top_k']
    )

    # Evaluación final
    print("\n=== Evaluación Final ===")
    model.eval()
    with torch.no_grad():
        total_mse = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in val_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs, _, _, _, _ = model(encoder_input_ids, decoder_input_ids)
            
            predictions = outputs.argmax(dim=-1)
            labels_flat = labels.view(-1)
            predictions_flat = predictions.view(-1)
            
            mask = labels_flat != -100
            labels_flat = labels_flat[mask]
            predictions_flat = predictions_flat[mask]
            
            # MSE
            labels_np = labels_flat.cpu().numpy()
            predictions_np = predictions_flat.cpu().numpy()
            batch_mse = mean_squared_error(labels_np, predictions_np)
            total_mse += batch_mse * len(labels_flat)
            
            # Accuracy
            correct = (predictions_flat == labels_flat).sum().item()
            total_correct += correct
            total_tokens += len(labels_flat)
            total_samples += 1

        # Calcular métricas finales
        avg_mse = total_mse / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        print("\nMétricas finales:")
        print(f"- MSE promedio: {avg_mse:.4f}")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- Total de muestras evaluadas: {total_samples}")
        print(f"- Total de tokens evaluados: {total_tokens}")

    # Limpiar
    monitor.remove_hooks()
    print("\nAnálisis completado.")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_perplexities': val_perplexities,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'final_metrics': {
            'mse': avg_mse,
            'accuracy': accuracy
        }
    }

def analyze_token_transformations_extended(model, tokenizer, prompt, train_dataloader, val_dataloader, 
                                            max_length=50, min_length=10, temperature=0.7, top_k=50):
    print("\n=== Análisis Extendido de Transformaciones de Tokens ===")
    
    # Análisis de transformación básico con la nueva implementación
    analyze_token_transformations(model, tokenizer, prompt, max_length, min_length, temperature, top_k)
    
    print("\n=== Análisis de Secuencias del Dataset ===")
    # Métricas de longitud de secuencia
    train_stats = {
        'seq_lens': [],
        'unique_tokens': set(),
        'token_freqs': {},
        'total_tokens': 0
    }
    val_stats = {
        'seq_lens': [],
        'unique_tokens': set(),
        'token_freqs': {},
        'total_tokens': 0
    }

    print("\nAnalizando conjunto de entrenamiento...")
    for batch in tqdm(train_dataloader):
        for seq in batch['input_ids']:
            seq_tokens = seq[seq != tokenizer.pad_token_id].tolist()
            train_stats['seq_lens'].append(len(seq_tokens))
            train_stats['unique_tokens'].update(seq_tokens)
            train_stats['total_tokens'] += len(seq_tokens)
            for token in seq_tokens:
                train_stats['token_freqs'][token] = train_stats['token_freqs'].get(token, 0) + 1

    print("\nAnalizando conjunto de validación...")
    for batch in tqdm(val_dataloader):
        for seq in batch['input_ids']:
            seq_tokens = seq[seq != tokenizer.pad_token_id].tolist()
            val_stats['seq_lens'].append(len(seq_tokens))
            val_stats['unique_tokens'].update(seq_tokens)
            val_stats['total_tokens'] += len(seq_tokens)
            for token in seq_tokens:
                val_stats['token_freqs'][token] = val_stats['token_freqs'].get(token, 0) + 1

    # Calcular y mostrar estadísticas
    print("\n=== Estadísticas de Secuencias ===")
    for split_name, stats in [("Entrenamiento", train_stats), ("Validación", val_stats)]:
        avg_len = np.mean(stats['seq_lens']) if stats['seq_lens'] else 0
        std_len = np.std(stats['seq_lens']) if stats['seq_lens'] else 0
        
        print(f"\nEstadísticas de {split_name}:")
        print(f"- Longitud promedio: {avg_len:.2f} ± {std_len:.2f}")
        print(f"- Longitud mínima: {min(stats['seq_lens']) if stats['seq_lens'] else 0}")
        print(f"- Longitud máxima: {max(stats['seq_lens']) if stats['seq_lens'] else 0}")
        print(f"- Tokens únicos: {len(stats['unique_tokens'])}")
        print(f"- Total de tokens: {stats['total_tokens']}")

        # Top 10 tokens más frecuentes
        top_tokens = sorted(stats['token_freqs'].items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 tokens más frecuentes:")
        for token_id, freq in top_tokens:
            token_text = tokenizer.decode([token_id])
            print(f"  {token_text}: {freq} veces ({freq/stats['total_tokens']*100:.2f}%)")

    # Análisis del prompt
    print("\n=== Análisis del Prompt ===")
    tokens = tokenizer.encode(prompt)
    unique_tokens = set(tokens)
    print(f"- Longitud del prompt: {len(tokens)}")
    print(f"- Tokens únicos en el prompt: {len(unique_tokens)}")
    print(f"- Tokens del prompt:")
    for token in tokens:
        print(f"  {token}: {tokenizer.decode([token])}")

    return train_stats, val_stats

def analyze_token_transformations(
    model, 
    tokenizer, 
    prompt, 
    max_length=50,
    min_length=10,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    length_penalty=1.0,
    num_beams=4,
    dynamic_temperature=True,
    dynamic_temp_factor=0.85,
    diversity_penalty=0.3,      # Penalización por diversidad entre beams
    min_tokens_per_beam=5,      # Mínimo de tokens por beam antes de podar
    adaptive_temp_threshold=0.1, # Umbral para temperatura adaptativa
    token_cooldown=3,           # Cooldown para tokens repetidos
    entropy_threshold=2.0,      # Umbral de entropía para diversidad
    batch_size=4,                # Procesar múltiples beams en paralelo
    cache_max_size=1000          # Tamaño máximo de la caché
):
    """
    Versión mejorada del analizador de transformaciones de tokens con optimizaciones adicionales.
    """

    # Validación de parámetros de entrada
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("`max_length` debe ser un entero positivo.")
    if not isinstance(min_length, int) or min_length < 0 or min_length > max_length:
        raise ValueError("`min_length` debe ser un entero no negativo y menor o igual a `max_length`.")
    if not (0 < temperature <= 1.0):
        raise ValueError("`temperature` debe estar en el rango (0, 1].")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("`top_k` debe ser un entero positivo.")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("`top_p` debe estar en el rango (0, 1].")
    if not (0.0 <= diversity_penalty < 1.0):
        raise ValueError("`diversity_penalty` debe estar en el rango [0, 1).")
    if not isinstance(num_beams, int) or num_beams <= 0:
        raise ValueError("`num_beams` debe ser un entero positivo.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("`batch_size` debe ser un entero positivo.")
    if batch_size > num_beams:
        batch_size = num_beams
    if not isinstance(cache_max_size, int) or cache_max_size <= 0:
        raise ValueError("`cache_max_size` debe ser un entero positivo.")

    print("\n=== Iniciando Generación de Tokens Avanzada ===")
    print(f"Prompt: {prompt}")
    
    # Configuración y logging mejorado
    config = {
        'max_length': max_length,
        'min_length': min_length,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'length_penalty': length_penalty,
        'num_beams': num_beams,
        'dynamic_temperature': dynamic_temperature,
        'dynamic_temp_factor': dynamic_temp_factor,
        'diversity_penalty': diversity_penalty,
        'min_tokens_per_beam': min_tokens_per_beam,
        'adaptive_temp_threshold': adaptive_temp_threshold,
        'token_cooldown': token_cooldown,
        'entropy_threshold': entropy_threshold,
        'batch_size': batch_size,
        'cache_max_size': cache_max_size
    }
    
    print("\nConfiguración:")
    for key, value in config.items():
        print(f"- {key}: {value}")

    device = next(model.parameters()).device
    print(f"\nDispositivo: {device}")

    # Definir tokens especiales a evitar, incluyendo todos los relevantes
    special_tokens_ids = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
        tokenizer.mask_token_id
    }

    # Inicialización
    beam_manager = BeamManager(num_beams, device, token_cooldown, diversity_penalty)
    cache = ComputationCache(max_size=cache_max_size)

    # Tokenización de entrada
    encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if encoder_input_ids.size(0) != 1:
        raise ValueError("La tokenización de entrada debe resultar en un tensor de batch size 1.")
    encoder_input_ids = encoder_input_ids.to(device)
    print(f"\n1. Tokenización de entrada:")
    print(f"   Forma de entrada codificada: {encoder_input_ids.shape}")
    print(f"   Entrada decodificada: {tokenizer.decode(encoder_input_ids[0])}")

    # Preparar entradas del modelo
    decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=torch.long)  # Shape: [1, 1]
    beam_sequences = [decoder_input_ids.clone() for _ in range(num_beams)]  # Lista de tensores

    # Función para calcular entropía usando distribuciones de PyTorch
    def calculate_entropy(probs):
        """Calcula la entropía de una distribución de probabilidad."""
        dist = torch.distributions.Categorical(probs)
        return dist.entropy().item()

    # Función para temperatura adaptativa con límite en el ajuste
    def get_adaptive_temperature(entropy, base_temp, max_allowed_temp_factor=2.0):
        """Ajusta la temperatura basado en la entropía de la distribución."""
        if entropy < adaptive_temp_threshold:
            adjustment = min(1 + (adaptive_temp_threshold - entropy), max_allowed_temp_factor)
            return base_temp * adjustment
        return base_temp

    model.eval()
    with torch.no_grad():
        # Codificación inicial
        encoder_output = model.encoder(encoder_input_ids)[0]
        print(f"\n2. Embedding de entrada:")
        print(f"   Forma del embedding: {encoder_output.shape}")
        print(f"   Estadísticas del embedding - Media: {encoder_output.mean().item():.4f}, Desv. Est.: {encoder_output.std().item():.4f}")

        for step in range(max_length):
            try:
                # Procesar beams en lotes para eficiencia
                for batch_start in range(0, num_beams, batch_size):
                    batch_end = min(batch_start + batch_size, num_beams)
                    current_batch_beam_indices = list(range(batch_start, batch_end))
                    active_beam_indices = [idx for idx in current_batch_beam_indices if idx not in beam_manager.finished_beams]
                    
                    if not active_beam_indices:
                        continue

                    # Preparar secuencias del batch
                    current_sequences = torch.stack([beam_sequences[idx] for idx in active_beam_indices])

                    # Generar clave de caché
                    cache_key = get_cache_key(current_sequences)
                    cached_result = cache.get(cache_key)

                    if cached_result is not None:
                        next_token_logits_batch = cached_result
                        print(f"\n3. Caché utilizada para el batch de beams {active_beam_indices}")
                    else:
                        # Generar logits
                        outputs = model(encoder_input_ids.repeat(len(active_beam_indices), 1), current_sequences)
                        next_token_logits_batch = outputs[:, -1, :]
                        cache.put(cache_key, next_token_logits_batch)
                        print(f"\n3. Logits generados y almacenados en caché para el batch de beams {active_beam_indices}")

                    for i, beam_idx in enumerate(active_beam_indices):
                        next_token_logits = next_token_logits_batch[i:i+1]  # Shape: [1, vocab_size]

                        # Aplicar penalización por diversidad
                        next_token_logits = beam_manager.calculate_diversity_penalty(next_token_logits, beam_idx)

                        # Enmascarar tokens especiales
                        for special_token_id in special_tokens_ids:
                            if special_token_id is not None:
                                next_token_logits[:, special_token_id] = float('-inf')
                        
                        # Si no ha alcanzado min_length, enmascarar EOS
                        if step < min_length:
                            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                                next_token_logits[:, tokenizer.eos_token_id] = float('-inf')

                        # Aplicar penalización por repetición
                        generated_tokens = set(beam_sequences[beam_idx].tolist())
                        for token_id in generated_tokens:
                            if token_id in special_tokens_ids:
                                continue  # Opcional: No penalizar tokens especiales
                            next_token_logits[:, token_id] /= repetition_penalty

                        # Calcular entropía y ajustar temperatura
                        probs = F.softmax(next_token_logits, dim=-1)
                        entropy = calculate_entropy(probs)
                        current_temp = temperature
                        
                        if dynamic_temperature:
                            progress_temp = temperature * (dynamic_temp_factor ** (step / max_length))
                            current_temp = get_adaptive_temperature(entropy, progress_temp)
                        
                        # Aplicar temperatura
                        next_token_logits = next_token_logits / current_temp

                        # Aplicar top-k
                        vocab_size = next_token_logits.size(-1)
                        effective_top_k = min(top_k, vocab_size)
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, k=effective_top_k, dim=-1)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                        # Aplicar top-p (nucleus sampling)
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = 0
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[:, indices_to_remove] = float('-inf')

                        # Calcular scores finales usando log_softmax
                        next_scores = F.log_softmax(next_token_logits, dim=-1)

                        # Aplicar length penalty
                        if length_penalty != 1.0:
                            length_pen = ((5 + step + 1) / 6) ** length_penalty
                            next_scores = next_scores / length_pen

                        # Seleccionar siguiente token
                        next_score, next_token = next_scores.max(dim=-1)

                        # Actualizar score
                        beam_manager.scores[beam_idx] = next_score

                        # Actualizar secuencia
                        new_sequence = torch.cat([beam_sequences[beam_idx], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                        beam_sequences[beam_idx] = new_sequence

                        # Actualizar historial de tokens con cooldown
                        beam_manager.update_token_history(beam_idx, next_token.item())

                        # Verificar finalización
                        if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id and step >= min_length:
                            beam_manager.finished_beams.add(beam_idx)
                            print(f"Beam {beam_idx} finalizado en el paso {step+1} con token EOS.")

            except AssertionError as ae:
                print(f"Error de aserción en el paso {step+1}: {str(ae)}")
                print(f"Shapes - decoder_input_ids: {decoder_input_ids.shape}, next_token: {next_token.shape}")
                raise ae
            except Exception as e:
                print(f"Error inesperado en el paso {step+1}: {str(e)}")
                traceback.print_exc()
                raise e

            # Verificar si todos los beams han terminado
            try:
                if len(beam_manager.finished_beams) == num_beams:
                    print(f"\nTodos los beams han finalizado en el paso {step+1}.")
                    break
            except Exception as e:
                print(f"Error al verificar beams finalizados: {str(e)}")
                raise e

            # Poda de beams de bajo rendimiento
            try:
                if step + 1 >= min_tokens_per_beam:
                    active_beams = num_beams - len(beam_manager.finished_beams)
                    if active_beams > 1:
                        scores = beam_manager.scores.clone()
                        scores[list(beam_manager.finished_beams)] = float('-inf')
                        worst_score = scores.topk(active_beams, largest=False)[0][-1]

                        for beam_idx in range(num_beams):
                            if (beam_idx not in beam_manager.finished_beams and 
                                beam_manager.scores[beam_idx] < worst_score and 
                                beam_sequences[beam_idx].size(1) >= min_tokens_per_beam):
                                beam_manager.finished_beams.add(beam_idx)
                                print(f"Beam {beam_idx} podado por bajo rendimiento en el paso {step+1}.")
            except Exception as e:
                print(f"Error al podar beams: {str(e)}")
                raise e

    # Seleccionar mejor secuencia, preferiblemente finalizada
    finished_beams = list(beam_manager.finished_beams)
    if finished_beams:
        finished_scores = beam_manager.scores[finished_beams]
        best_finished_idx = finished_scores.argmax().item()
        best_beam_idx = finished_beams[best_finished_idx]
    else:
        best_beam_idx = beam_manager.scores.argmax().item()

    best_sequence = beam_sequences[best_beam_idx]

    # Análisis final
    print("\n=== Estadísticas Finales ===")
    print(f"Longitud de secuencia: {best_sequence.size(1)}")
    print(f"Score final: {beam_manager.scores[best_beam_idx].item():.4f}")
    print(f"Beams completados: {len(beam_manager.finished_beams)}")

    # Generar texto
    generated_text = tokenizer.decode(best_sequence[0], skip_special_tokens=True)
    print("\nTexto generado:")
    print(generated_text)

    return best_sequence[0], beam_manager.scores[best_beam_idx].item(), {
        'sequence_length': best_sequence.size(1),
        'completed_beams': len(beam_manager.finished_beams),
        'final_score': beam_manager.scores[best_beam_idx].item()
    }

class BeamManager:
    def __init__(self, num_beams, device, token_cooldown, diversity_penalty):
        self.num_beams = num_beams
        self.device = device
        self.sequences = [deque(maxlen=token_cooldown) for _ in range(num_beams)]
        self.scores = torch.zeros(num_beams, device=device)
        self.finished_beams = set()
        self.diversity_penalty = diversity_penalty

    def update_token_history(self, beam_idx, token):
        """Actualiza el historial de tokens con sistema de cooldown."""
        self.sequences[beam_idx].append(token)

    def get_active_tokens(self, beam_idx):
        """Retorna tokens actualmente en cooldown."""
        return set(self.sequences[beam_idx])

    def calculate_diversity_penalty(self, logits, beam_idx):
        """Calcula penalización por diversidad basada en historial."""
        active_tokens = self.get_active_tokens(beam_idx)
        penalty = torch.ones_like(logits)
        if active_tokens:
            penalty[list(active_tokens)] *= (1.0 - self.diversity_penalty)
        return logits * penalty

class ComputationCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Eliminar entrada más antigua
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

def get_cache_key(sequences):
    """Genera una clave de caché robusta usando SHA256."""
    return hashlib.sha256(sequences.cpu().numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    results = main_analysis()
