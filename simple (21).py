import math
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func

# Descargar todos los recursos de NLTK
nltk.download('all')

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Constantes globales
EMBED_DIM = 512
NUM_LAYERS = 12
NUM_HEADS = 8
FF_HIDDEN_DIM = 1024
NUM_EXPERTS = 8
EXPERT_DIM = 512
MAX_LENGTH = 8192
WINDOW_SIZE = 512
COMPRESSION_RATIO = 0.5
BATCH_SIZE = 4
NUM_EPOCHS = 10
ACCUMULATION_STEPS = 4
TOP_K = 4        # Número inicial de expertos a preseleccionar
DYNAMIC_K = True # Activar el ajuste dinámico de K
NUM_KV_GROUPS = 2  # Añadido: Número de grupos para Group Query Attention

# Implementación de Hooks para Monitoreo
class ActivationMonitor:
    def __init__(self, model):
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.register_hooks(model)

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                handle = module.register_forward_hook(self.save_activation(name))
                handle_grad = module.register_full_backward_hook(self.save_gradient(name))
                self.handles.append(handle)
                self.handles.append(handle_grad)

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
            # Verificar rango de activaciones
            if not torch.isfinite(output).all():
                print(f"Activaciones no finitas en {name}")
        return hook

    def save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
            # Verificar rango de gradientes
            if not torch.isfinite(grad_output[0]).all():
                print(f"Gradientes no finitos en {name}")
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

class GradientMonitor:
    def __init__(self, model):
        self.gradients = {}
        self.register_hooks(model)

    def register_hooks(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.save_gradient(name))

    def save_gradient(self, name):
        def hook(grad):
            if grad is not None:
                self.gradients[name] = grad.detach()
                grad_norm = grad.norm().item()
                if not torch.isfinite(grad).all():
                    print(f"Gradiente no finito en {name} con norma {grad_norm}")
                elif grad_norm > 10:  # Umbral de clipping adicional
                    print(f"Gradiente excesivamente grande en {name} con norma {grad_norm}")
        return hook

# Función auxiliar para obtener la siguiente potencia de dos
def next_power_of_two(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()
class LiquidEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=2048, base_compression_ratio=0.5, 
                 min_compression_ratio=0.1, epsilon=1e-8):
        super(LiquidEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.base_compression_ratio = base_compression_ratio
        self.min_compression_ratio = min_compression_ratio
        self.epsilon = epsilon
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Capas convolucionales
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Projection head ligero (opcional, para mejor estructura semántica)
        self.proj_head = nn.Linear(embed_dim, embed_dim)
        
        # Capas principales
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(0.15)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Inicialización original
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.position_embedding.weight, gain=1.0)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)
        # Nueva inicialización para proj_head
        nn.init.xavier_normal_(self.proj_head.weight, gain=1.0)
        nn.init.zeros_(self.proj_head.bias)

    def compute_loss(self, x_ifft, recon_target, mask):

        # 1. Normalización base
        x_norm = F.normalize(x_ifft, p=2, dim=-1)
        target_norm = F.normalize(recon_target, p=2, dim=-1)
        
        # 2. Similitud coseno con epsilon
        cos_sim = torch.sum(x_norm * target_norm, dim=-1)
        cos_sim = torch.clamp(cos_sim, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        
        # 3. Pérdida L1 suavizada
        l1_loss = F.smooth_l1_loss(x_ifft, recon_target, reduction='none')
        
        # 4. Componente semántico (opcional, para mejor preservación de significado)
        proj_x = self.proj_head(x_ifft)
        proj_target = self.proj_head(recon_target)
        semantic_loss = F.mse_loss(
            F.normalize(proj_x, p=2, dim=-1),
            F.normalize(proj_target, p=2, dim=-1),
            reduction='none'
        ).mean(dim=-1)
        
        # 5. Combinación de pérdidas
        combined_loss = (
            0.6 * (1.0 - cos_sim).mean() +     # Similitud direccional
            0.2 * l1_loss.mean(dim=-1) +       # Reconstrucción detallada
            0.2 * semantic_loss                 # Preservación semántica
        )
        
        # 6. Aplicar máscara y normalizar
        masked_loss = (combined_loss * mask).sum() / (mask.sum() + self.epsilon)
        
        return masked_loss

    def forward(self, x):

        batch_size, seq_length = x.size()  # Definir antes del try-except
        try:
            # Manejo de secuencia
            seq_length = min(seq_length, self.position_embedding.num_embeddings)
            x = x[:, :seq_length]

            # Embeddings con mejor manejo de posiciones
            positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            x = self.token_embedding(x) + self.position_embedding(positions)

            # Proceso convolucional mejorado
            x_conv = x.transpose(1, 2)
            x_conv = F.gelu(self.conv1(x_conv))
            x_conv = F.gelu(self.conv2(x_conv))
            x = x_conv.transpose(1, 2)

            # Normalización y regularización
            x = self.layer_norm(x)
            x = self.dropout(x)

            # Preparación para FFT (sin cambios en la lógica core)
            padded_length = next_power_of_two(seq_length)
            padding = padded_length - seq_length
            if padding > 0:
                x = F.pad(x, (0, 0, 0, padding))

            # FFT y análisis de complejidad (manteniendo lógica original)
            x_fft = torch.fft.fft(x, dim=1)
            magnitude = torch.abs(x_fft)
            complexity = torch.mean(
                (magnitude > 0.1 * torch.max(magnitude, dim=1, keepdim=True)[0]).float(),
                dim=(1, 2)
            )

            # Compresión adaptativa (preservada)
            compression_ratio = torch.clamp(
                self.base_compression_ratio * (1 - complexity),
                min=self.min_compression_ratio,
                max=1.0
            )

            # Proceso de compresión (mantenido)
            N = (compression_ratio * seq_length).long()
            N = torch.clamp(N, min=1, max=seq_length)
            max_N = N.max().item()

            # Tensores comprimidos con mejor manejo de memoria
            x_fft_compressed = torch.zeros(
                (batch_size, max_N, self.embed_dim),
                dtype=torch.complex64,
                device=x.device
            )
            mask = torch.zeros(batch_size, seq_length, dtype=torch.float32, device=x.device)

            # Aplicar compresión (lógica original)
            for i, n in enumerate(N):
                n = n.item()
                x_fft_compressed[i, :n] = x_fft[i, :n]
                mask[i, :n] = 1.0

            # Reconstrucción con mejor estabilidad
            x_ifft = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
            x_ifft = self.proj(x_ifft)
            x_ifft = self.layer_norm(x_ifft)

            # Target mejorado
            recon_target = self.proj(
                torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
            )

            # Calcular pérdida mejorada
            loss = self.compute_loss(
                x_ifft[:, :seq_length],
                recon_target[:, :seq_length],
                mask[:, :seq_length]
            )

            return x_ifft[:, :seq_length], loss

        except Exception as e:
            print(f"Error en LiquidEmbedding forward: {str(e)}")
            return (
                torch.zeros((batch_size, seq_length, self.embed_dim), device=x.device),
                torch.tensor(0.0, device=x.device)
            )

def apply_rotary_pos_emb(q, k, sin, cos):
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]

    q_rot = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
    k_rot = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)
    return q_rot, k_rot
class EnhancedLocalAttentionWithGQA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=256, bidirectional=True, 
                 dropout=0.12, num_kv_groups=2, epsilon=1e-8, max_seq_length=8192):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.head_dim = embed_dim // num_heads
        self.num_kv_groups = num_kv_groups
        self.epsilon = epsilon
        self.max_seq_length = max_seq_length
        
        # Proyecciones separadas para differential attention
        self.q_proj1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj1 = nn.Linear(embed_dim, embed_dim // (num_heads // num_kv_groups), bias=False)
        self.k_proj2 = nn.Linear(embed_dim, embed_dim // (num_heads // num_kv_groups), bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // (num_heads // num_kv_groups), bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scaler para precisión mixta
        self.grad_scaler = GradScaler()

        # Lambda como parámetro aprendible con inicialización específica
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))
        
        # Buffers para RoPE
        self.register_buffer(
            "sin",
            torch.zeros(max_seq_length, self.head_dim // 2),
            persistent=False
        )
        self.register_buffer(
            "cos",
            torch.zeros(max_seq_length, self.head_dim // 2),
            persistent=False
        )
        
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2).float() * 
                           (-math.log(10000.0) / self.head_dim))
        self.sin[:, :] = torch.sin(position * div_term)
        self.cos[:, :] = torch.cos(position * div_term)
        
        self._init_weights()

    def _init_weights(self):
        # Inicialización específica para transformers con precisión mixta
        for param in self.parameters():
            if param.ndim > 1:
                nn.init.xavier_uniform_(param, gain=1/math.sqrt(2))
        
        # Inicializar lambdas cerca de los valores objetivo
        with torch.no_grad():
            self.lambda_q1.data.fill_(0.1)
            self.lambda_k1.data.fill_(0.1)
            self.lambda_q2.data.fill_(-0.1)
            self.lambda_k2.data.fill_(-0.1)

    def compute_lambda(self):
        # Cálculo estable de lambda para precisión mixta
        with torch.cuda.amp.autocast():
            lambda_value = torch.exp(self.lambda_q1 @ self.lambda_k1) - \
                         torch.exp(self.lambda_q2 @ self.lambda_k2) + self.lambda_init
            return lambda_value.clamp(0.1, 0.9)  # Mantener en rango estable

    def forward(self, x):
        with torch.cuda.amp.autocast():
            try:
                B, L, C = x.shape
                pad_l = (self.window_size - L % self.window_size) % self.window_size
                x_padded = nn.functional.pad(x, (0, 0, 0, pad_l))
                _, L_padded, _ = x_padded.shape

                # Proyecciones con precisión mixta
                q1 = self.q_proj1(x_padded).reshape(B, L_padded, self.num_heads, self.head_dim)
                q2 = self.q_proj2(x_padded).reshape(B, L_padded, self.num_heads, self.head_dim)
                k1 = self.k_proj1(x_padded).reshape(B, L_padded, self.num_kv_groups, self.head_dim)
                k2 = self.k_proj2(x_padded).reshape(B, L_padded, self.num_kv_groups, self.head_dim)
                v = self.v_proj(x_padded).reshape(B, L_padded, self.num_kv_groups, self.head_dim)

                # Preparar para atención
                q1 = q1.permute(0, 2, 1, 3).contiguous()
                q2 = q2.permute(0, 2, 1, 3).contiguous()
                k1 = k1.permute(0, 2, 1, 3).contiguous()
                k2 = k2.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()

                # Convertir a float16/bfloat16 para FlashAttention
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                q1, q2 = q1.to(dtype), q2.to(dtype)
                k1, k2 = k1.to(dtype), k2.to(dtype)
                v = v.to(dtype)

                # Aplicar RoPE
                if L_padded > self.max_seq_length:
                    raise ValueError(f"Secuencia excede longitud máxima RoPE: {self.max_seq_length}")
                
                sin = self.sin[:L_padded, :].unsqueeze(0).unsqueeze(0).to(dtype)
                cos = self.cos[:L_padded, :].unsqueeze(0).unsqueeze(0).to(dtype)
                
                q1, k1 = apply_rotary_pos_emb(q1, k1, sin, cos)
                q2, k2 = apply_rotary_pos_emb(q2, k2, sin, cos)

                # Procesar por ventanas
                causal = not self.bidirectional
                num_windows = (L_padded - self.window_size) // (self.window_size // 2) + 1
                attn_outputs = []
                
                lambda_value = self.compute_lambda().to(dtype)
                
                for i in range(num_windows):
                    start_idx = i * (self.window_size // 2)
                    end_idx = start_idx + self.window_size
                    
                    if end_idx <= L_padded:
                        # Extraer ventanas
                        q1_window = q1[..., start_idx:end_idx, :]
                        q2_window = q2[..., start_idx:end_idx, :]
                        k1_window = k1[..., start_idx:end_idx, :]
                        k2_window = k2[..., start_idx:end_idx, :]
                        v_window = v[..., start_idx:end_idx, :]

                        # Repetir KV para heads
                        k1_window = k1_window.repeat(1, self.num_heads // self.num_kv_groups, 1, 1)
                        k2_window = k2_window.repeat(1, self.num_heads // self.num_kv_groups, 1, 1)
                        v_window = v_window.repeat(1, self.num_heads // self.num_kv_groups, 1, 1)

                        try:
                            # FlashAttention con precisión mixta
                            with torch.cuda.amp.autocast():
                                attn1 = flash_attn_func(
                                    q1_window, k1_window, v_window,
                                    dropout_p=self.dropout.p if self.training else 0.0,
                                    causal=causal,
                                    softmax_scale=1/math.sqrt(self.head_dim)
                                )
                                attn2 = flash_attn_func(
                                    q2_window, k2_window, v_window,
                                    dropout_p=self.dropout.p if self.training else 0.0,
                                    causal=causal,
                                    softmax_scale=1/math.sqrt(self.head_dim)
                                )
                                
                                # Atención diferencial
                                attn_output = attn1 - lambda_value * attn2

                            if not torch.isfinite(attn_output).all():
                                raise ValueError(f"Salida no finita en ventana {i}")
                                
                        except Exception as e:
                            print(f"Error en flash_attn_func ventana {i}: {str(e)}")
                            attn_output = torch.zeros_like(q1_window)
                            
                        attn_outputs.append(attn_output)

                # Procesar salida
                attn_output = torch.cat(attn_outputs, dim=2)
                attn_output = attn_output.transpose(1, 2).contiguous()
                
                # Volver a float32 para el resto de la red
                attn_output = attn_output.float()
                attn_output = attn_output.view(B, -1, self.embed_dim)
                
                if attn_output.size(1) > L:
                    attn_output = attn_output[:, :L, :]
                    
                # Proyección final
                attn_output = self.out(attn_output)

                if not torch.isfinite(attn_output).all():
                    raise ValueError("Salida de atención no finita")

                return attn_output

            except Exception as e:
                print(f"Error en forward pass: {str(e)}")
                return torch.zeros((B, L, self.embed_dim), device=x.device, dtype=torch.float32)

# Clase MoELayer con inicialización mejorada
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim, dropout=0.12, entropy_weight=0.05, top_k=2, dynamic_k=False, max_usage_ratio=0.4, epsilon=1e-8):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dynamic_k = dynamic_k
        self.epsilon = epsilon  # Añadido epsilon para estabilidad
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.entropy_weight = entropy_weight
        self.max_usage_ratio = max_usage_ratio
        self.expert_usage_counter = None

        # Inicialización de expertos
        for expert in self.experts:
            nn.init.orthogonal_(expert.weight, gain=math.sqrt(2))
            nn.init.zeros_(expert.bias)
            
        # Inicialización del gate con valores pequeños para comenzar con routing casi uniforme
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Verificación de Finiteza en gate_probs
        if not torch.isfinite(gate_probs).all():
            raise ValueError("gate_probs contiene valores no finitos")

        # Regularización de entropía
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + self.epsilon), dim=-1).mean()
        entropy_loss = self.entropy_weight * entropy

        # Ajuste dinámico de K
        if self.dynamic_k:
            complexity = entropy.detach().item()
            K = max(1, min(self.num_experts, int(self.top_k * (1 + complexity))))
        else:
            K = self.top_k

        topk_probs, topk_indices = torch.topk(gate_probs, K, dim=-1)

        # Inicializar o reiniciar el contador de uso de expertos
        if self.expert_usage_counter is None:
            self.expert_usage_counter = torch.zeros(self.num_experts, device=x.device)
        else:
            self.expert_usage_counter = self.expert_usage_counter.to(x.device)

        expert_outputs = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device, dtype=x.dtype)

        for k in range(K):
            expert_idx = topk_indices[:, k]
            # Crear una máscara para seleccionar las posiciones que usan este experto
            mask = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1) == expert_idx.unsqueeze(1)
            mask = mask.any(dim=1)
            selected_x = x_flat[mask]

            if selected_x.size(0) > 0:
                # Seleccionar el experto de manera eficiente
                unique_experts = expert_idx[mask].unique()
                for expert in unique_experts:
                    expert_mask = expert_idx[mask] == expert
                    inputs = selected_x[expert_mask]

                    try:
                        output = self.dropout(self.experts[expert](inputs))
                        # **Verificación de Finiteza en la salida de los expertos**
                        if not torch.isfinite(output).all():
                            raise ValueError(f"Salida del experto {expert} contiene valores no finitos")
                    except Exception as e:
                        print(f"Error en la salida del experto {expert}: {str(e)}")
                        output = torch.zeros_like(inputs)

                    expert_outputs[mask][expert_mask] += output * topk_probs[:, k][mask][expert_mask].unsqueeze(1)
                    # Actualizar el contador de uso de expertos
                    self.expert_usage_counter[expert] += inputs.size(0)

        # Calcular la penalización por uso excesivo
        usage_ratios = self.expert_usage_counter / (batch_size * seq_length + self.epsilon)  # Añadido epsilon para evitar división por cero
        overuse_penalty = torch.sum(F.relu(usage_ratios - self.max_usage_ratio))

        # **Verificación de Finiteza en overuse_penalty**
        if not torch.isfinite(overuse_penalty).all():
            raise ValueError("overuse_penalty contiene valores no finitos")

        output = expert_outputs.view(batch_size, seq_length, -1)

        # **Verificación de Finiteza en la salida final de MoELayer**
        if not torch.isfinite(output).all():
            raise ValueError("Salida de MoELayer contiene valores no finitos")

        return output, entropy_loss + overuse_penalty

    def get_expert_usage_stats(self):
        if self.expert_usage_counter is None:
            return None
        total_usage = self.expert_usage_counter.sum().item()
        if total_usage == 0:
            return [0.0] * self.num_experts
        usage_percentages = (self.expert_usage_counter / (total_usage + self.epsilon) * 100).tolist()  # Añadido epsilon
        return usage_percentages

# Clase DeformableConv1d con inicialización mejorada
class DeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, epsilon=1e-8):
        super(DeformableConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding  # Debe ser un entero
        self.stride = stride
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = epsilon  # Añadido epsilon para estabilidad

        # Convolución para generar los offsets
        self.offset_conv = nn.Conv1d(
            in_channels,
            2 * kernel_size,  # Para desplazamientos en la dimensión de longitud
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )

        # Inicialización adecuada de los pesos
        nn.init.kaiming_normal_(self.offset_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.offset_conv.bias)

        # Convolución principal ajustada para recibir C * kernel_size canales
        self.conv = nn.Conv1d(
            in_channels * kernel_size,  # Cambiado de in_channels a in_channels * kernel_size
            out_channels,
            kernel_size=1,  # Kernel size ajustado para operar sobre los canales
            stride=1,       # Stride ajustado
            padding=0,      # Padding ajustado
            dilation=1,     # Dilation ajustado
            bias=bias
        )

        # Inicialización adecuada de los pesos
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        with autocast(enabled=True):
            try:
                offsets = self.offset_conv(x)  # [N, 2 * kernel_size, L_out]
                
                # **Verificación de Finiteza de Offsets**
                if not torch.isfinite(offsets).all():
                    raise ValueError("Offsets contienen valores no finitos")
                
                N, _, L_out = offsets.size()
                offsets = offsets.view(N, self.kernel_size, 2, L_out)  # [N, kernel_size, 2, L_out]
                offsets = offsets.permute(0, 3, 1, 2)  # [N, L_out, kernel_size, 2]
                x_padded = F.pad(x, (self.padding, self.padding))  # [N, C, L + 2 * padding]
                device = x.device
                dtype = x.dtype
                base_grid = torch.arange(0, x_padded.size(2), device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)  # [1, L_padded, 1]
                base_grid = base_grid.repeat(N, 1, self.kernel_size)  # [N, L_padded, kernel_size]
                grid = base_grid[:, self.padding:x_padded.size(2)-self.padding, :] + offsets[..., 0]  # [N, L_out, kernel_size]
                grid = grid.clamp(0, x_padded.size(2) - 1)
                
                left = grid.floor().long()  # [N, L_out, kernel_size]
                right = (left + 1).clamp(max=x_padded.size(2) - 1)  # [N, L_out, kernel_size]
                
                # **Verificaciones de Límites de Índices**
                if not ((left >= 0).all() and (left < x_padded.size(2)).all()):
                    raise ValueError("Índices 'left' fuera de límites")
                if not ((right >= 0).all() and (right < x_padded.size(2)).all()):
                    raise ValueError("Índices 'right' fuera de límites")

                alpha = grid - left.float()  # [N, L_out, kernel_size]
                left = left.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]
                right = right.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]
                x_left = torch.gather(x_padded, 2, left)  # [N, C, L_out * kernel_size]
                x_right = torch.gather(x_padded, 2, right)  # [N, C, L_out * kernel_size]
                x_left = x_left.view(N, self.in_channels, L_out, self.kernel_size)  # [N, C, L_out, kernel_size]
                x_right = x_right.view(N, self.in_channels, L_out, self.kernel_size)  # [N, C, L_out, kernel_size]
                alpha = alpha.view(N, 1, L_out, self.kernel_size)  # [N, 1, L_out, kernel_size]
                x_deform = (1 - alpha) * x_left + alpha * x_right  # [N, C, L_out, kernel_size]
                x_deform = x_deform.permute(0, 3, 2, 1).contiguous().view(N, self.in_channels * self.kernel_size, L_out)  # [N, C * kernel_size, L_out]
                out = self.conv(x_deform)  # [N, out_channels, L_out]

                # **Verificación de Finiteza de la Salida**
                if not torch.isfinite(out).all():
                    raise ValueError("Salida de deform_conv contiene valores no finitos")
            except Exception as e:
                print(f"Error en DeformableConv1d forward: {str(e)}")
                # Retornar valores seguros en caso de error
                out = torch.zeros((x.size(0), self.out_channels, x.size(-1)), device=x.device, dtype=x.dtype)
        
        return out

# Clase OptimizedGatedConvolution con inicialización mejorada
class OptimizedGatedConvolution(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.12, epsilon=1e-8):
        super(OptimizedGatedConvolution, self).__init__()
        self.epsilon = epsilon
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = nn.Dropout(dropout)
        padding = (kernel_size - 1) * dilation // 2
        self.deform_conv = DeformableConv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            epsilon=epsilon
        )
        
        # Inicialización de pesos
        nn.init.kaiming_normal_(self.deform_conv.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.deform_conv.conv.bias)
        nn.init.kaiming_normal_(self.deform_conv.offset_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.deform_conv.offset_conv.bias)
        
    def forward(self, x):
        def conv_function(x):
            # Asegurar que x requiere gradientes
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
                
            with autocast(enabled=True):
                try:
                    x = x.transpose(1, 2)
                    conv_out = self.deform_conv(x)
                    
                    if not torch.isfinite(conv_out).all():
                        raise ValueError("conv_out contiene valores no finitos después de deform_conv")
                    
                    main, gate = conv_out.chunk(2, dim=1)
                    main = F.gelu(main)
                    gate = torch.sigmoid(gate)
                    gated_out = main * gate
                    
                    if not torch.isfinite(gated_out).all():
                        raise ValueError("gated_out contiene valores no finitos después de GELU y Sigmoid")
                    
                    gated_out = self.dropout(gated_out)
                    mean = gated_out.mean(dim=1, keepdim=True)
                    var = gated_out.var(dim=1, keepdim=True, unbiased=False)
                    gated_out = (gated_out - mean) / torch.sqrt(var + self.epsilon)
                    
                    if not torch.isfinite(gated_out).all():
                        raise ValueError("gated_out contiene valores no finitos después de la normalización")
                    
                    return gated_out.transpose(1, 2)
                except Exception as e:
                    print(f"Error en OptimizedGatedConvolution forward: {str(e)}")
                    return torch.zeros_like(x).transpose(1, 2)
        
        # Asegurar que al menos un tensor requiere gradientes antes de checkpoint
        if isinstance(x, torch.Tensor) and not x.requires_grad:
            x = x.detach().requires_grad_(True)
            
        return checkpoint(conv_function, x, use_reentrant=True)

# Clase EnhancedLSTM con inicialización mejorada
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.12, epsilon=1e-8):
        super(EnhancedLSTM, self).__init__()
        self.epsilon = epsilon  # Añadido epsilon para estabilidad
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)  # Agregar dropout
        
        # LSTM estándar de PyTorch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)  # Agregar dropout al LSTM
        
        # Capa de salida adicional con GELU
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )

        # Inicialización de pesos del LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)
        
        # Inicialización de la capa de salida
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, hidden=None):
        try:
            # Pasar a través del LSTM
            lstm_out, hidden = self.lstm(x, hidden)
            lstm_out = self.dropout(lstm_out)  # Aplicar dropout

            # Aplicar la capa de salida con conexión residual
            output = self.output_layer(lstm_out) + x

            # **Verificación de Finiteza en la salida del LSTM**
            if not torch.isfinite(output).all():
                raise ValueError("Salida de EnhancedLSTM contiene valores no finitos")
        except Exception as e:
            print(f"Error en EnhancedLSTM forward: {str(e)}")
            output = torch.zeros_like(x)
            hidden = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])) if hidden is not None else None

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device))
        return hidden

# Clase ImprovedTransformerBlock con inicialización mejorada
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, window_size=256, 
                 bidirectional=True, dropout=0.12, entropy_weight=0.05, top_k=2, dynamic_k=False, 
                 num_kv_groups=2, epsilon=1e-8):
        super(ImprovedTransformerBlock, self).__init__()
        self.epsilon = epsilon
        
        # Layer normalization antes de la atención (arquitectura Pre-LN)
        # Usar la nueva atención diferencial
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attention = EnhancedLocalAttentionWithGQA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            bidirectional=bidirectional,
            dropout=dropout,
            num_kv_groups=num_kv_groups,
            epsilon=epsilon
        )
        self.dropout1 = nn.Dropout(dropout)
        # Layer normalization antes de la convolución
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dilated_conv = OptimizedGatedConvolution(
            channels=embed_dim, 
            kernel_size=3, 
            dilation=2, 
            dropout=dropout, 
            epsilon=epsilon
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer normalization antes de MoE
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.moe = MoELayer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            dropout=dropout,
            entropy_weight=entropy_weight,
            top_k=top_k,
            dynamic_k=dynamic_k,
            epsilon=epsilon
        )
        self.dropout3 = nn.Dropout(dropout)
        
        # Red feedforward modificada con normalización intermedia
        self.ff_layer = nn.ModuleList([
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.LayerNorm(ff_hidden_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        ])
        
        # Inicializar pesos con mejor escalado
        self._initialize_weights()
        
        # Valor de clipping de gradientes
        self.grad_clip = 1.0

    def _initialize_weights(self):
        for layer in self.ff_layer:
            if isinstance(layer, nn.Linear):
                # Uso de Kaiming para mejor flujo de gradiente
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
        # Escalar la inicialización de la última capa para prevenir explosiones de gradientes
        if isinstance(self.ff_layer[-2], nn.Linear):
            nn.init.kaiming_normal_(self.ff_layer[-2].weight, mode='fan_in', nonlinearity='linear')
            self.ff_layer[-2].weight.data *= 0.1
            if self.ff_layer[-2].bias is not None:
                nn.init.zeros_(self.ff_layer[-2].bias)

    def forward(self, x):
        with autocast(enabled=True):
            try:
                # Bloque de atención con conexión residual
                residual = x
                x = self.norm1(x)
                attn_out = self.attention(x)
                attn_out = self.dropout1(attn_out)
                x = residual + attn_out
                
                # Bloque de convolución con conexión residual
                residual = x
                x = self.norm2(x)
                conv_out = self.dilated_conv(x)
                conv_out = self.dropout2(conv_out)
                x = residual + conv_out
                
                # Bloque MoE con conexión residual
                residual = x
                x = self.norm3(x)
                moe_output, entropy_loss = self.moe(x)
                moe_output = self.dropout3(moe_output)
                x = residual + moe_output
                
                # Bloque feedforward con conexión residual y verificaciones intermedias
                residual = x
                for i, layer in enumerate(self.ff_layer):
                    x = layer(x)
                    # Añadir verificaciones de estabilidad después de operaciones clave
                    if i in [0, 4]:  # Después de capas lineales
                        x = torch.clamp(x, min=-100, max=100)  # Prevenir valores extremos
                    if i == 2:  # Después de GELU
                        x = x.nan_to_num(0.0)  # Reemplazar NaNs con ceros
                    # Verificar que la salida sea finita después de cada operación
                    if not torch.isfinite(x).all():
                        raise ValueError(f"Valores no finitos detectados después de la capa {i} en ff_layer")
                
                x = residual + x
                
                # Verificación de estabilidad final
                if not torch.isfinite(x).all():
                    x = torch.nan_to_num(x, nan=0.0, posinf=100, neginf=-100)

            except Exception as e:
                print(f"Error en ImprovedTransformerBlock forward: {str(e)}")
                x = residual + x  # Mantener la residual en caso de error
                entropy_loss = torch.tensor(0.0, device=x.device)

        return x, entropy_loss

    def clip_gradients(self):
        # Clip gradients for all parameters in the block
        for p in self.parameters():
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_(p, self.grad_clip)
class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256, num_experts=2, expert_dim=256, entropy_weight=0.05, top_k=2, dynamic_k=False, num_kv_groups=2, dropout=0.12, epsilon=1e-8):
        super(BidirectionalEncoder, self).__init__()
        self.epsilon = epsilon  # Añadido epsilon para estabilidad
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, base_compression_ratio=compression_ratio, epsilon=epsilon)
        self.layers = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim, 
                window_size=window_size, 
                bidirectional=True, 
                dropout=dropout,  # Usar el dropout especificado
                entropy_weight=entropy_weight, 
                top_k=top_k, 
                dynamic_k=dynamic_k,
                num_kv_groups=num_kv_groups,
                epsilon=epsilon  # Pasar epsilon
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)  # Agregar dropout

        # Inicialización de layer_norm y dropout si es necesario
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(self, x):
        with autocast(enabled=True):
            try:
                x, recon_loss = self.embedding(x)
                x = self.dropout(x)
                total_entropy_loss = recon_loss.new_zeros(1)  # Initialize as tensor
                for layer in self.layers:
                    x, recon_loss_layer = layer(x)
                    total_entropy_loss += recon_loss_layer
                x = self.layer_norm(x)
                # **Verificación de Finiteza después de layer_norm**
                if not torch.isfinite(x).all():
                    raise ValueError("Salida después de layer_norm contiene valores no finitos")
                return x, recon_loss, total_entropy_loss
            except Exception as e:
                print(f"Error en BidirectionalEncoder forward: {str(e)}")
                x = torch.zeros_like(x)
                recon_loss = torch.tensor(0.0, device=x.device)
                total_entropy_loss = torch.tensor(0.0, device=x.device)
                return x, recon_loss, total_entropy_loss

class LiquidFoundationModelOptimized(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=12, num_heads=8, ff_hidden_dim=1024,
                 num_experts=8, expert_dim=512, max_length=8192, window_size=512, compression_ratio=0.5, 
                 entropy_weight=0.15, top_k=4, dynamic_k=True, lstm_hidden_size=256, lstm_num_layers=2, 
                 dropout=0.1, epsilon=1e-8, num_kv_groups=2):
        super(LiquidFoundationModelOptimized, self).__init__()
        self.epsilon = epsilon
        self.generation_mode = False  # Flag para modo de generación
        
        # Componentes principales
        self.encoder = BidirectionalEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            compression_ratio=compression_ratio,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            window_size=window_size,
            num_experts=num_experts,
            expert_dim=expert_dim,
            entropy_weight=entropy_weight,
            top_k=top_k,
            dynamic_k=dynamic_k,
            dropout=dropout,
            num_kv_groups=num_kv_groups,
            epsilon=epsilon
        )
        
        self.decoder_embedding = LiquidEmbedding(
            vocab_size, 
            embed_dim, 
            max_length, 
            base_compression_ratio=0.5, 
            min_compression_ratio=0.1,
            epsilon=epsilon
        )
        
        self.decoder_layers = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim, 
                window_size=window_size, 
                bidirectional=False, 
                dropout=dropout,
                entropy_weight=entropy_weight, 
                top_k=top_k, 
                dynamic_k=dynamic_k,
                num_kv_groups=num_kv_groups,
                epsilon=epsilon
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Memoria externa
        self.external_memory = EnhancedLSTM(
            embed_dim, 
            lstm_hidden_size, 
            num_layers=lstm_num_layers, 
            dropout=dropout, 
            epsilon=epsilon
        )
        
        # Parámetros de configuración
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        
        # Inicialización de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa los pesos del modelo usando técnicas modernas."""
        # Inicialización de la capa de salida
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)
        
        # Verificar que todos los parámetros están inicializados
        for name, param in self.named_parameters():
            if param.data.dim() > 1:
                if 'weight' in name:
                    if 'norm' in name:
                        nn.init.ones_(param)
                    else:
                        nn.init.xavier_uniform_(param, gain=1.0)
            else:
                if 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, encoder_input_ids, decoder_input_ids):
        with torch.cuda.amp.autocast(enabled=True):
            try:
                # 1. Validación de entrada
                if encoder_input_ids.dim() != 2 or decoder_input_ids.dim() != 2:
                    raise ValueError(f"Las entradas deben ser 2D. Shapes: encoder={encoder_input_ids.shape}, decoder={decoder_input_ids.shape}")

                # 2. Asegurar que los batch sizes coinciden
                batch_size = encoder_input_ids.size(0)
                if decoder_input_ids.size(0) != batch_size:
                    raise ValueError(f"Batch sizes no coinciden: encoder={batch_size}, decoder={decoder_input_ids.size(0)}")

                # 3. Truncar secuencias si exceden max_length
                encoder_input_ids = encoder_input_ids[:, :self.max_length]
                decoder_input_ids = decoder_input_ids[:, :self.max_length]

                # 4. Procesamiento del encoder con manejo de errores
                encoder_output, recon_loss_enc, entropy_loss_enc = self.encoder(encoder_input_ids)

                # 5. Embeddings del decoder con padding seguro
                decoder_embeddings, recon_loss_dec = self.decoder_embedding(decoder_input_ids)
                decoder_embeddings = self.dropout(decoder_embeddings)

                # 6. Inicialización segura del estado oculto LSTM
                hidden = self.external_memory.init_hidden(batch_size)

                # 7. Procesamiento por capas con manejo de errores
                total_entropy_loss_dec = torch.tensor(0.0, device=decoder_input_ids.device)
                for layer in self.decoder_layers:
                    layer_out, entropy_loss = layer(decoder_embeddings)
                    decoder_embeddings = layer_out
                    total_entropy_loss_dec += entropy_loss

                    # Actualizar LSTM
                    decoder_embeddings, hidden = self.external_memory(decoder_embeddings, hidden)

                # 8. Normalización y capa de salida con validaciones
                decoder_embeddings = self.layer_norm(decoder_embeddings)
                logits = self.output_layer(decoder_embeddings)

                # Verificar y corregir NaNs/Infs
                if not torch.isfinite(logits).all():
                    print("Detectados valores no finitos en logits")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                # Asegurar dimensiones correctas
                logits = logits[:, :min(2048, logits.size(1)), :]

                return (
                    logits,
                    recon_loss_enc,
                    recon_loss_dec,
                    entropy_loss_enc,
                    total_entropy_loss_dec
                )

            except Exception as e:
                print(f"Error general en forward pass: {str(e)}")
                # Inferir batch_size desde encoder_input_ids o decoder_input_ids
                batch_size = encoder_input_ids.size(0) if encoder_input_ids.dim() >= 1 else 1
                return (
                    torch.zeros((batch_size, min(2048, decoder_input_ids.size(1)), self.output_layer.out_features), device=decoder_input_ids.device),
                    torch.tensor(0.0, device=decoder_input_ids.device),
                    torch.tensor(0.0, device=decoder_input_ids.device),
                    torch.tensor(0.0, device=decoder_input_ids.device),
                    torch.tensor(0.0, device=decoder_input_ids.device)
                )
    
    def generate(self, encoder_input_ids, tokenizer, max_length=2048, start_token_id=None):

        if start_token_id is None:
            raise ValueError("start_token_id debe ser proporcionado para la generación")
        
        # Activar modo generación
        self.set_generation_mode(True)
        
        try:
            device = encoder_input_ids.device
            batch_size = encoder_input_ids.size(0)
            decoder_input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)  # [batch_size, 1]
            generated_ids = [[] for _ in range(batch_size)]
            
            for step in range(max_length):
                logits, _, _, _, _ = self.forward(encoder_input_ids, decoder_input_ids)  # logits: [batch_size, seq_length, vocab_size]
                next_token_logits = logits[:, -1, :] / self.temperature  # [batch_size, vocab_size]
                probabilities = F.softmax(next_token_logits, dim=-1)  # [batch_size, vocab_size]
                next_tokens = torch.multinomial(probabilities, num_samples=1)  # [batch_size, 1]
                
                # Actualizar decoder_input_ids
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)  # [batch_size, step+2]
                
                # Guardar tokens generados
                for i in range(batch_size):
                    generated_ids[i].append(next_tokens[i].item())
                
                # Verificar tokens de fin
                if (next_tokens == tokenizer.eos_token_id).all():
                    print(f"Todos los beams han generado el token de fin en el paso {step+1}")
                    break
            
            # Decodificar textos generados
            generated_texts = [
                tokenizer.decode(decoder_input_ids[i].tolist(), skip_special_tokens=True)
                for i in range(batch_size)
            ]
            
            print("\nTexto generado:")
            for idx, text in enumerate(generated_texts):
                print(f"Beam {idx + 1}: {text}")
            
            return decoder_input_ids, generated_texts
            
        except Exception as e:
            print(f"Error en la función de generación: {str(e)}")
            return None, None
            
        finally:
            # Desactivar modo generación al finalizar
            self.set_generation_mode(False)
print("Modelo cargado correctamente con mejoras de estabilidad numérica.")
print("A")
