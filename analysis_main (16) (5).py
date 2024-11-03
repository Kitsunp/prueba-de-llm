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
import traceback  # Añadido para manejar excepciones

from datasets import load_dataset
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate import meteor_score  # Añadido para METEOR
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, f1_score  # f1_score añadido
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
from transformers import LEDTokenizer, LEDForConditionalGeneration, LongformerTokenizer
from transformers import LEDTokenizer, LEDModel

from simple import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,

)
from simple import (
    ActivationMonitor,
    LiquidFoundationModelOptimized,
    MoELayer,
)

# Añadir importación para ROUGE
from rouge_score import rouge_scorer

# Configuración global (simplificada)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # Probabilidad del modelo para la clase correcta
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def verify_no_nans(batch):

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

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError(f"Batch contiene valores no finitos en {key} después de la limpieza.")
    return batch
def prepare_data(max_samples=None, val_size=0.1, max_length=16384):  # Ajusta max_length según el modelo LED
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
        print("Cargando tokenizer LED...")
        tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
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

        def tokenize_data(examples, tokenizer, max_length=16384, verbose=True):  # Ajustar max_length
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
                    try:
                        sep_pos = ids.index(tokenizer.sep_token_id)
                    except ValueError:
                        # Asignar sep_pos de manera predeterminada si no se encuentra el sep_token
                        sep_pos = len(ids) // 2  # Por ejemplo, dividir la secuencia a la mitad
                        print(f"Advertencia: sep_token_id no encontrado. Asignando sep_pos a {sep_pos}.")

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
print("G")
def train_one_batch(batch, model, ce_criterion, focal_criterion, optimizer, scaler, device, tokenizer, accumulation_steps, metrics_tracker):
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
                # Pérdidas
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

                # Backward pass
                scaled_loss = total_loss / accumulation_steps
                scaler.scale(scaled_loss).backward()

                # Predictions
                predictions = valid_logits.argmax(dim=-1)
                pred_tokens = predictions.cpu().tolist()
                true_tokens = valid_labels.cpu().tolist()

                # Actualizar métricas de loss
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

                # Sequence accuracy
                batch_predictions = outputs.argmax(dim=-1)
                correct_sequences = (batch_predictions == labels).all(dim=1).sum().item()
                metrics_tracker['correct_sequences'] += correct_sequences
                metrics_tracker['total_sequences'] += labels.size(0)

                # Top-k accuracy
                for k in [1, 3, 5, 10]:
                    top_k_preds = torch.topk(valid_logits, k, dim=-1).indices
                    correct_k = sum(valid_labels[i] in top_k_preds[i] for i in range(len(valid_labels)))
                    metrics_tracker[f'top_k_accuracy_{k}'] += float(correct_k)

                # Vocabulary y diversidad
                metrics_tracker['pred_tokens'].extend(pred_tokens)
                metrics_tracker['true_tokens'].extend(true_tokens)
                metrics_tracker['unique_unigrams'].update(pred_tokens)
                metrics_tracker['vocab_size'] = len(metrics_tracker['unique_unigrams'])

                # Bigramas
                for i in range(len(pred_tokens) - 1):
                    metrics_tracker['unique_bigrams'].add((pred_tokens[i], pred_tokens[i + 1]))

                # BLEU score
                decoded_preds = tokenizer.batch_decode(predictions.unsqueeze(1), skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(valid_labels.unsqueeze(1), skip_special_tokens=True)
                
                # BLEU score - Versión corregida
                try:
                    decoded_preds = tokenizer.batch_decode(predictions.unsqueeze(1), skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(valid_labels.unsqueeze(1), skip_special_tokens=True)
                    
                    smoothing = SmoothingFunction().method1
                    for n in range(1, 5):
                        weights = tuple([1.0/n] * n + [0.0] * (4-n))
                        batch_bleu = np.mean([
                            sentence_bleu(
                                [ref.split()],  # Referencias deben estar en una lista
                                hyp.split(),
                                weights=weights,
                                smoothing_function=smoothing
                            )
                            for ref, hyp in zip(decoded_labels, decoded_preds)
                        ])
                        # Acumulamos el promedio del batch
                        metrics_tracker[f'bleu_{n}'] += batch_bleu
                        metrics_tracker[f'bleu_batches'] = metrics_tracker.get('bleu_batches', 0) + 1

                except Exception as e:
                    print(f"Error calculando BLEU: {str(e)}")

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
        traceback.print_exc()
        raise e


def calculate_rouge(predictions, references):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Promediar las puntuaciones
    for key in rouge_scores:
        rouge_scores[key] /= len(predictions)
    
    return rouge_scores

# Esta es la definición que realmente se usa y necesita ser actualizada
def calculate_meteor(predictions, references):  # <- ACTUALIZAR ESTA

    meteor_scores = []
    for pred, ref in zip(predictions, references):
        try:
            # Tokenizar las secuencias
            pred_tokens = word_tokenize(str(pred))
            ref_tokens = word_tokenize(str(ref))
            
            # Calcular METEOR score
            score = meteor_score.single_meteor_score(ref_tokens, pred_tokens)
            meteor_scores.append(score)
        except Exception as e:
            print(f"Error calculando METEOR para predicción: {pred[:50]}... y referencia: {ref[:50]}...")
            print(f"Error específico: {str(e)}")
            continue
    
    return np.mean(meteor_scores) if meteor_scores else 0.0
def evaluate(model, data_loader, ce_criterion, focal_criterion, device, tokenizer):

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
        # Nuevas métricas
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'meteor': 0.0,
    }
    predictions = []
    references = []
    
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
                    predictions_ids = valid_logits.argmax(dim=-1)
                    metrics['correct_tokens'] += (predictions_ids == valid_labels).sum().item()

                    batch_predictions = outputs.argmax(dim=-1)
                    metrics['correct_sequences'] += (batch_predictions == labels).all(dim=1).sum().item()
                    metrics['total_sequences'] += labels.size(0)

                    # Top-k accuracy
                    for k in [1, 3, 5, 10]:
                        top_k_preds = torch.topk(valid_logits, k, dim=-1).indices
                        correct_k = sum(valid_labels[i] in top_k_preds[i] for i in range(len(valid_labels)))
                        metrics[f'top_k_accuracy_{k}'] += float(correct_k)

                    # Diversidad y vocabulario
                    pred_tokens = predictions_ids.cpu().tolist()
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

                    # Acumular predicciones y referencias para ROUGE y METEOR
                    decoded_preds_batch = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
                    decoded_refs_batch = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
                    predictions.extend(decoded_preds_batch)
                    references.extend(decoded_refs_batch)
    
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
            # Nuevas métricas calculadas al final
            'rouge1': calculate_rouge(predictions, references)['rouge1'],
            'rouge2': calculate_rouge(predictions, references)['rouge2'],
            'rougeL': calculate_rouge(predictions, references)['rougeL'],
            'meteor': calculate_meteor(predictions, references),
        }

        # Calcular BLEU scores si no se ha hecho previamente
        if predictions and references:
            try:
                smoothing = SmoothingFunction().method1
                bleu_scores = {f'bleu_{n}': 0.0 for n in range(1, 5)}
                for n in range(1, 5):
                    weights = tuple([1.0/n] * n + [0.0] * (4-n))
                    bleu_scores[f'bleu_{n}'] = np.mean([
                        sentence_bleu([ref.split()], hyp.split(), weights=weights, smoothing_function=smoothing)
                        for ref, hyp in zip(references, predictions)
                    ])
                final_metrics.update(bleu_scores)
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
            # Nuevas métricas
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'meteor': 0.0,
        }

    return final_metrics
print("G")
def log_metrics(writer, train_metrics, val_metrics, epoch):

    # Logging a TensorBoard
    # Métricas básicas
    writer.add_scalar('Loss/train', train_metrics['total_loss'] / train_metrics['valid_tokens'], epoch)
    writer.add_scalar('Loss/val', val_metrics['total_loss'], epoch)
    writer.add_scalar('Perplexity/val', val_metrics['perplexity'], epoch)

    # Accuracy
    writer.add_scalar('Accuracy/token', train_metrics['correct_tokens'] / train_metrics['valid_tokens'], epoch)
    writer.add_scalar('Accuracy/sequence', train_metrics['correct_sequences'] / train_metrics['total_sequences'] if train_metrics['total_sequences'] > 0 else 0.0, epoch)
    for k in [1, 3, 5, 10]:
        writer.add_scalar(f'Accuracy/top_{k}', train_metrics[f'top_k_accuracy_{k}'] / train_metrics['valid_tokens'], epoch)

    # Diversidad
    writer.add_scalar('Diversity/distinct_1', len(train_metrics['unique_unigrams']) / len(train_metrics['pred_tokens']) if train_metrics['pred_tokens'] else 0.0, epoch)
    writer.add_scalar('Diversity/distinct_2', len(train_metrics['unique_bigrams']) / max(1, len(train_metrics['pred_tokens']) - 1) if train_metrics['pred_tokens'] else 0.0, epoch)

    # Fluidez
    writer.add_scalar('Fluency/avg_sequence_length', float(np.mean(train_metrics['sequence_lengths'])) if train_metrics['sequence_lengths'] else 0.0, epoch)

    # BLEU
    for n in range(1, 5):
        writer.add_scalar(f'BLEU/bleu_{n}', train_metrics[f'bleu_{n}'], epoch)

    # Vocabulario
    writer.add_scalar('Vocabulary/vocab_size', train_metrics['vocab_size'], epoch)

    # Nuevas Métricas
    writer.add_scalar('ROUGE/ROUGE-1', val_metrics['rouge1'], epoch)
    writer.add_scalar('ROUGE/ROUGE-2', val_metrics['rouge2'], epoch)
    writer.add_scalar('ROUGE/ROUGE-L', val_metrics['rougeL'], epoch)
    writer.add_scalar('METEOR/METEOR', val_metrics['meteor'], epoch)

    # Logging en consola
    print("\n--- Métricas de Entrenamiento ---")
    print(f"Total Loss: {train_metrics['total_loss'] / train_metrics['valid_tokens']:.4f}")
    print(f"  - CrossEntropy Loss: {train_metrics['ce_loss'] / train_metrics['valid_tokens']:.4f}")
    print(f"  - Focal Loss: {train_metrics['focal_loss'] / train_metrics['valid_tokens']:.4f}")

    print("\n--- Métricas de Validación ---")
    print(f"Total Loss: {val_metrics['total_loss']:.4f}")
    print(f"  - CrossEntropy Loss: {val_metrics['ce_loss']:.4f}")
    print(f"  - Focal Loss: {val_metrics['focal_loss']:.4f}")

    print("\n--- Métricas de Accuracy ---")
    print(f"Token Accuracy: {train_metrics['correct_tokens'] / train_metrics['valid_tokens']:.4f}")
    print(f"Sequence Accuracy: {train_metrics['correct_sequences'] / train_metrics['total_sequences']:.4f}")
    for k in [1, 3, 5, 10]:
        print(f"Top-{k} Accuracy: {train_metrics[f'top_k_accuracy_{k}'] / train_metrics['valid_tokens']:.4f}")

    print("\n--- Métricas de Diversidad ---")
    print(f"Distinct-1: {len(train_metrics['unique_unigrams']) / len(train_metrics['pred_tokens']) if train_metrics['pred_tokens'] else 0.0:.4f}")
    print(f"Distinct-2: {len(train_metrics['unique_bigrams']) / max(1, len(train_metrics['pred_tokens']) - 1) if train_metrics['pred_tokens'] else 0.0:.4f}")

    print("\n--- Métricas de Fluidez ---")
    print(f"Avg Sequence Length: {float(np.mean(train_metrics['sequence_lengths'])) if train_metrics['sequence_lengths'] else 0.0:.2f}")

    # BLEU Scores
    num_batches = train_metrics.get('bleu_batches', 1)  # Evitar división por cero
    print("\n--- BLEU Scores ---")
    for n in range(1, 5):
        # Calculamos el promedio dividiendo por el número de batches
        avg_bleu = train_metrics[f'bleu_{n}'] / num_batches
        writer.add_scalar(f'BLEU/bleu_{n}', avg_bleu, epoch)
        print(f"BLEU-{n}: {avg_bleu:.4f}")

    print("\n--- Métricas de Vocabulario ---")
    print(f"Vocab Size: {train_metrics['vocab_size']}")

    print("\n--- Nuevas Métricas ---")
    print(f"ROUGE-1: {val_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {val_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {val_metrics['rougeL']:.4f}")
    print(f"METEOR: {val_metrics['meteor']:.4f}")

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
    tokenizer,
    num_epochs, 
    accumulation_steps=10, 
    monitor=None, 
    debug_loss_components=True
):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    epoch = 0

    # Lists for storing metric history
    train_losses = []
    val_losses = []
    val_perplexities = []

    # Definimos metrics_tracker como una variable mutable (diccionario)
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
        'sequence_lengths': [],
        'perplexities': [],
        'top_k_accuracy_1': 0.0,
        'top_k_accuracy_3': 0.0,
        'top_k_accuracy_5': 0.0,
        'top_k_accuracy_10': 0.0,
        # BLEU scores y contadores
        'bleu_1': 0.0,
        'bleu_2': 0.0,
        'bleu_3': 0.0,
        'bleu_4': 0.0,
        'bleu_batches': 0,  # Contador para promediar        
        'bleu_1_samples': 0,
        'bleu_2_samples': 0,
        'bleu_3_samples': 0,
        'bleu_4_samples': 0,
        # Vocab size
        'vocab_size': 0,
        # Otras métricas
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'meteor': 0.0,
    }

    def reset_metrics():
        """Resets all metrics to their initial values."""
        metrics_tracker.clear()
        metrics_tracker.update({
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
            # BLEU scores y contadores
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'bleu_1_samples': 0,
            'bleu_2_samples': 0,
            'bleu_3_samples': 0,
            'bleu_4_samples': 0,
            # Vocab size
            'vocab_size': 0,
            # Otras métricas
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'meteor': 0.0,
        })

    def calculate_metrics(model, data_loader, is_training=True):
        """Calculates all metrics on a dataset."""
        return evaluate(model, data_loader, ce_criterion, focal_criterion, device, tokenizer)

    try:
        # Verify model parameters
        print("Checking model parameters that don't require gradients...")
        frozen_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_params.append(name)
        if frozen_params:
            print("Warning! The following parameters don't require gradients:")
            for name in frozen_params:
                print(f" - {name}")
            raise ValueError("Some model parameters don't require gradients.")
        else:
            print("All model parameters require gradients. Proceeding with training.")

        while epoch < num_epochs:
            model.train()
            print(f"\n--- Epoch {epoch + 1} ---")
            reset_metrics()  # Reset metrics al inicio de cada época

            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in loop:
                try:
                    train_one_batch(
                        batch, model, ce_criterion, focal_criterion, 
                        optimizer, scaler, device, tokenizer,
                        accumulation_steps, metrics_tracker
                    )

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    # Actualizar la barra de progreso
                    if metrics_tracker['valid_tokens'] > 0:
                        current_loss = metrics_tracker['total_loss'] / metrics_tracker['valid_tokens']
                        loop.set_postfix(loss=f"{current_loss:.4f}")

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue

            try:
                torch.cuda.empty_cache()
                gc.collect()

                # Evaluación en el conjunto de validación
                val_metrics = calculate_metrics(model, val_loader, is_training=False)

                # Acumular pérdidas y perplejidades para graficar
                val_losses.append(val_metrics['total_loss'])
                val_perplexities.append(val_metrics['perplexity'])

                print(f"\nEpoch {epoch + 1} completed. Evaluating model...")

                # Loguear métricas
                log_metrics(writer, metrics_tracker, val_metrics, epoch + 1)

                # Guardar el mejor modelo basado en la pérdida de validación
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    no_improve = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'metrics': {
                            'train': metrics_tracker,
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
                    print("\nSaved new best model!")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("\nEarly stopping triggered")
                        break

                del val_metrics
                torch.cuda.empty_cache()
                gc.collect()

                epoch += 1
                print(f"\nEpoch {epoch} completed. Batches processed: {len(train_loader)}")

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break

    finally:
        writer.close()
        if monitor:
            monitor.remove_hooks()

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Val')
            plt.title('Loss Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(val_perplexities)
            plt.title('Perplexity Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.yscale('log')

            plt.tight_layout()
            plt.savefig('training_evolution.png')
            plt.close()

            print("\nFinal visualizations saved in 'training_evolution.png'")
        except Exception as e:
            print(f"Error generating final visualizations: {str(e)}")
            traceback.print_exc()

    return train_losses, val_losses, val_perplexities
def main_analysis():
    print("\n=== Iniciando Análisis Principal ===")
    
    # Configuración inicial
    config = {
        'max_samples': 3000,
        'val_size': 0.1,
        'max_length': 400,  # Ajustado para LED
        'batch_size': 4,
        'num_epochs': 5,  # Ajusta según tus necesidades
        'lr': 0.0001,
        'temperature': 0.7,
        'top_k': 50,
        'min_length': 10,
        'diversity_penalty': 0.1,  # Reducido de 0.3
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

    # Crear el modelo con el vocab_size correcto
    model = LiquidFoundationModelOptimized(vocab_size=VOCAB_SIZE).to(device)

    # Verificar el tamaño de la capa de salida
    print(f"Shape de output_layer.weight: {model.output_layer.weight.shape}")
    print(f"out_features de output_layer: {model.output_layer.out_features}")

    # Verificar si la capa de salida tiene el tamaño correcto
    expected_vocab_size = len(tokenizer)  # 50261
    actual_out_features = model.output_layer.out_features

    if actual_out_features == expected_vocab_size:
        print("✅ La capa de salida tiene el tamaño correcto.")
    else:
        print(f"❌ Error: La capa de salida tiene {actual_out_features} out_features, pero se esperaba {expected_vocab_size}.")
        # Ajustar la capa de salida si es necesario
        model.output_layer = nn.Linear(embed_dim, expected_vocab_size)
        print(f"✅ Capa de salida ajustada a {expected_vocab_size} out_features.")

    # Verificar tokens especiales
    print(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"SEP Token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Criterios de pérdida
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    focal_criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100)
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = GradScaler()
    monitor = ActivationMonitor(model)  # Asegúrate de que esta clase esté bien definida

    # Configuración de entrenamiento
    print("\nEstadísticas de entrenamiento:")
    print(f"- Batches por época: {len(train_loader)}")
    print(f"- Total épocas objetivo: {config['num_epochs']}")
    print(f"- Intervalo de evaluación: al final de cada época")

    # Training
    print("\n=== Starting Training ===")
    train_losses, val_losses, val_perplexities = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        ce_criterion=ce_criterion,
        focal_criterion=focal_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        tokenizer=tokenizer,  # Pasar el tokenizador a train_model
        num_epochs=config['num_epochs'],
        accumulation_steps=ACCUMULATION_STEPS,
        monitor=monitor,
        debug_loss_components=False
    )

    # ... (Resto de la función permanece igual)


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
    
    # Generación de texto a partir del prompt
    generated_sequence, score, stats = analyze_token_transformations(
        model, tokenizer, prompt, max_length, min_length, temperature, top_k
    )
    
    if generated_sequence is None:
        print("Error durante la generación de tokens.")
        return {}, {}
    
    # Análisis adicional de las secuencias del dataset
    print("\n=== Análisis de Secuencias del Dataset ===")
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
    for batch in tqdm(train_dataloader, desc="Analizando entrenamiento"):
        for seq in batch['input_ids']:
            seq_tokens = seq[seq != tokenizer.pad_token_id].tolist()
            train_stats['seq_lens'].append(len(seq_tokens))
            train_stats['unique_tokens'].update(seq_tokens)
            train_stats['total_tokens'] += len(seq_tokens)
            for token in seq_tokens:
                train_stats['token_freqs'][token] = train_stats['token_freqs'].get(token, 0) + 1

    print("\nAnalizando conjunto de validación...")
    for batch in tqdm(val_dataloader, desc="Analizando validación"):
        for seq in batch['input_ids']:
            seq_tokens = seq[seq != tokenizer.pad_token_id].tolist()
            val_stats['seq_lens'].append(len(seq_tokens))
            val_stats['unique_tokens'].update(seq_tokens)
            val_stats['total_tokens'] += len(seq_tokens)
            for token in seq_tokens:
                val_stats['token_freqs'][token] = val_stats['token_freqs'].get(token, 0) + 1

    # Calcular y mostrar estadísticas
    print("\n=== Estadísticas de Secuencias ===")
    for split_name, stats_dict in [("Entrenamiento", train_stats), ("Validación", val_stats)]:
        avg_len = np.mean(stats_dict['seq_lens']) if stats_dict['seq_lens'] else 0
        std_len = np.std(stats_dict['seq_lens']) if stats_dict['seq_lens'] else 0
        
        print(f"\nEstadísticas de {split_name}:")
        print(f"- Longitud promedio: {avg_len:.2f} ± {std_len:.2f}")
        print(f"- Longitud mínima: {min(stats_dict['seq_lens']) if stats_dict['seq_lens'] else 0}")
        print(f"- Longitud máxima: {max(stats_dict['seq_lens']) if stats_dict['seq_lens'] else 0}")
        print(f"- Tokens únicos: {len(stats_dict['unique_tokens'])}")
        print(f"- Total de tokens: {stats_dict['total_tokens']}")

        # Top 10 tokens más frecuentes
        top_tokens = sorted(stats_dict['token_freqs'].items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 tokens más frecuentes:")
        for token_id, freq in top_tokens:
            token_text = tokenizer.decode([token_id])
            print(f"  {token_text}: {freq} veces ({freq/stats_dict['total_tokens']*100:.2f}%)")

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
    diversity_penalty=0.3,      
    min_tokens_per_beam=5,      
    adaptive_temp_threshold=0.1, 
    token_cooldown=3,           
    entropy_threshold=2.0,      
    batch_size=4,               
    cache_max_size=1000         
):

    # Validación de parámetros
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("`max_length` debe ser un entero positivo.")
    if not isinstance(min_length, int) or min_length < 0 or min_length > max_length:
        raise ValueError("`min_length` debe ser un entero no negativo y menor o igual a `max_length`.")
    if not (0 < temperature <= 1.0):
        raise ValueError("`temperature` debe estar en el rango (0, 1].")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("`top_k` debe ser un entero positivo.")
    if batch_size > num_beams:
        batch_size = num_beams  # Ajustar el batch_size si es mayor que num_beams

    print("\n=== Iniciando Generación de Tokens Avanzada ===")
    print(f"Prompt: {prompt}")
    
    # Configuración y logging
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

    # Tokens especiales a evitar
    special_tokens_ids = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
        tokenizer.mask_token_id
    }
    special_tokens_ids = {t for t in special_tokens_ids if t is not None}

    # Inicialización de BeamManager y ComputationCache
    beam_manager = BeamManager(num_beams, device, token_cooldown, diversity_penalty)
    cache = ComputationCache(max_size=cache_max_size)
    # Tokenización de entrada
    encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n1. Tokenización de entrada:")
    print(f"   Forma de entrada codificada: {encoder_input_ids.shape}")
    print(f"   Entrada decodificada: {tokenizer.decode(encoder_input_ids[0])}")
    
    # Preparar decoder_input_ids con token BOS
    decoder_input_ids = torch.full((num_beams, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)  # [num_beams, 1]
    beam_sequences = [decoder_input_ids[i].clone() for i in range(num_beams)]  # Lista de tensores individuales por beam
    
    model.eval()
    with torch.no_grad():
        try:
            # Codificación inicial a través del encoder
            encoder_output, recon_loss_enc, entropy_loss_enc = model.encoder(encoder_input_ids.repeat(num_beams, 1))
            
            print(f"\n2. Embedding de entrada:")
            print(f"   Forma del embedding: {encoder_output.shape}")
            print(f"   Estadísticas del embedding - Media: {encoder_output.mean().item():.4f}, "
                  f"Desv. Est.: {encoder_output.std().item():.4f}")

            # Generación iterativa
            for step in range(max_length):
                # Preparar secuencias del batch
                current_sequences = torch.stack(beam_sequences)  # [num_beams, seq_length]
                
                # Procesar a través del modelo
                logits, _, _, _, _ = model(encoder_input_ids.repeat(num_beams, 1), current_sequences)  # logits: [num_beams, seq_length, vocab_size]
                
                # Obtener los logits del último token generado
                next_token_logits = logits[:, -1, :] / temperature  # [num_beams, vocab_size]
                
                # Aplicar repetición de penalización
                if repetition_penalty != 1.0:
                    for beam_idx in range(num_beams):
                        for token in beam_sequences[beam_idx]:
                            if token.item() in special_tokens_ids:
                                continue
                            next_token_logits[beam_idx, token] /= repetition_penalty
                
                # Aplicar penalización de diversidad directamente a los logits
                for beam_idx in range(num_beams):
                    next_token_logits[beam_idx] = beam_manager.apply_diversity_penalty(beam_idx, next_token_logits[beam_idx])
                
                # Aplicar top-k y top-p filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                
                # Calcular probabilidades
                probabilities = F.softmax(next_token_logits, dim=-1)  # [num_beams, vocab_size]
                
                # Seleccionar siguiente token usando multinomial sampling
                next_tokens = torch.multinomial(probabilities, num_samples=1)  # [num_beams, 1]
                
                # Actualizar secuencias y scores
                for beam_idx in range(num_beams):
                    next_token = next_tokens[beam_idx].item()
                    token_text = tokenizer.decode([next_token])
                    print(f"Beam {beam_idx} generó token: {token_text} (ID: {next_token})")

                    beam_sequences[beam_idx] = torch.cat([beam_sequences[beam_idx], next_tokens[beam_idx]], dim=-1)
                    beam_manager.update_token_history(beam_idx, next_token)
                    
                    # Calcular log-probabilidad y actualizar score
                    log_prob = torch.log(probabilities[beam_idx, next_token] + 1e-10).item()  # Añadir epsilon para estabilidad
                    beam_manager.update_scores(beam_idx, log_prob)
                
                # Verificar tokens de fin (EOS)
                eos_mask = next_tokens.squeeze(-1) == tokenizer.eos_token_id
                for beam_idx in range(num_beams):
                    if eos_mask[beam_idx] and beam_idx not in beam_manager.finished_beams:
                        beam_manager.finished_beams.add(beam_idx)
                        print(f"Beam {beam_idx} finalizado en paso {step+1}")
                
                # Salir si todos los beams han finalizado
                if len(beam_manager.finished_beams) == num_beams:
                    print(f"\nTodos los beams han finalizado en paso {step+1}")
                    break

            # Seleccionar la mejor secuencia basada en los scores
            if beam_manager.finished_beams:
                best_beam_idx = max(
                    beam_manager.finished_beams,
                    key=lambda idx: beam_manager.scores[idx].item()
                )
            else:
                best_beam_idx = beam_manager.scores.argmax().item()
            
            best_sequence = beam_sequences[best_beam_idx]
            best_score = beam_manager.scores[best_beam_idx].item()
            
            # Generar estadísticas finales
            statistics = {
                'sequence_length': best_sequence.size(0),
                'completed_beams': len(beam_manager.finished_beams),
                'final_score': best_score
            }
            
            # Decodificar y mostrar texto generado
            generated_text = tokenizer.decode(best_sequence, skip_special_tokens=True)
            print("\nTexto generado:")
            print(generated_text)

            print("\n=== Estadísticas Finales ===")
            print(f"Longitud de secuencia: {best_sequence.size(0)}")
            print(f"Score final: {best_score:.4f}")
            print(f"Beams completados: {len(beam_manager.finished_beams)}")

            return best_sequence, best_score, statistics

        except Exception as e:
            print(f"Error en el proceso de generación: {str(e)}")
            traceback.print_exc()
            return None, 0.0, {}

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):

    top_k = min(top_k, logits.size(-1))  # Asegurar que top_k no exceda el tamaño del vocabulario
    if top_k > 0:
        # Mantener solo los top_k tokens y eliminar el resto
        topk_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_values[:, -1].unsqueeze(-1)
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Crear máscara para tokens que exceden el umbral de probabilidad acumulativa
        sorted_indices_to_remove = cumulative_probs > top_p

        # Mover el umbral un paso hacia la derecha para incluir al primer token que excede el umbral
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0  # Nunca eliminar el primer token

        # Crear una máscara de eliminación en el orden original
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # Aplicar la máscara
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

class BeamManager:
    def __init__(self, num_beams, device, token_cooldown, diversity_penalty):
        self.num_beams = num_beams
        self.device = device
        self.token_cooldown = token_cooldown
        self.diversity_penalty = diversity_penalty
        self.sequences = [deque(maxlen=token_cooldown) for _ in range(num_beams)]
        self.scores = torch.zeros(num_beams, device=device)
        self.finished_beams = set()

    def update_token_history(self, beam_idx, token):
        self.sequences[beam_idx].append(token)

    def get_active_tokens(self, beam_idx):
        return set(self.sequences[beam_idx])

    def apply_diversity_penalty(self, beam_idx, logits):

        active_tokens = self.get_active_tokens(beam_idx)
        if active_tokens:
            for token in active_tokens:
                if 0 <= token < logits.size(0):
                    logits[token] *= (1.0 - self.diversity_penalty)
                else:
                    print(f"Advertencia: Token ID {token} fuera de rango para penalización de diversidad.")
        return logits

    def update_scores(self, beam_idx, log_prob):

        self.scores[beam_idx] += log_prob

class ComputationCache:
    def __init__(self, max_size=1000):

        self.cache = {}
        self.max_size = max_size

    def get(self, key):

        return self.cache.get(key)

    def put(self, key, value):

        if len(self.cache) >= self.max_size:
            # Eliminar la entrada más antigua
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

def get_cache_key(sequences):
    """Genera una clave de caché robusta usando SHA256."""  
    return hashlib.sha256(sequences.cpu().numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    results = main_analysis()
