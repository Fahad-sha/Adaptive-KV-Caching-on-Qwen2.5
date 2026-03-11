import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import gradio as gr
import matplotlib.pyplot as plt
import time

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading Model on {DEVICE}...")
# Load Model Globally to avoid reloading on every message
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    attn_implementation="eager" # Required for attention-based eviction
)
if DEVICE == "cpu":
    model.to(DEVICE)
model.eval()

# --- Helper Functions from Original Code ---

def get_cache_memory_mb(past_key_values):
    if past_key_values is None:
        return 0.0
    total_bytes = 0
    if hasattr(past_key_values, 'layers') and past_key_values.layers is not None:
        for layer in past_key_values.layers:
            for attr_name in ['key', 'value', 'keys', 'values', 'k', 'v', 'key_states', 'value_states']:
                if hasattr(layer, attr_name):
                    tensor = getattr(layer, attr_name)
                    if tensor is not None and hasattr(tensor, 'nelement'):
                        total_bytes += tensor.element_size() * tensor.nelement()
    elif hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
        for k_layer, v_layer in zip(past_key_values.key_cache, past_key_values.value_cache):
            total_bytes += k_layer.element_size() * k_layer.nelement()
            total_bytes += v_layer.element_size() * v_layer.nelement()
    elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k_layer, v_layer = layer[0], layer[1]
                total_bytes += k_layer.element_size() * k_layer.nelement()
                total_bytes += v_layer.element_size() * v_layer.nelement()
    return total_bytes / (1024 * 1024)

def get_cache_seq_len(past_key_values):
    if past_key_values is None: return 0
    if hasattr(past_key_values, 'get_seq_length') and callable(past_key_values.get_seq_length):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, 'layers') and past_key_values.layers is not None:
        if len(past_key_values.layers) > 0:
            layer = past_key_values.layers[0]
            if hasattr(layer, 'key') and layer.key is not None:
                return layer.key.shape[2]
    return 0

# --- Core Generation Logic with Eviction ---

def generate_with_eviction(prompt, max_cache_size, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    past_key_values = None
    generated_ids = []
    
    # Tracking for Visualization
    cache_history = []
    seq_len_history = []
    eviction_events = [] # Store step index where eviction happened
    step_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                return_dict=True,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            attentions = outputs.attentions
            
            # 1. Track Memory
            cache_mb = get_cache_memory_mb(past_key_values)
            current_seq = get_cache_seq_len(past_key_values)
            cache_history.append(cache_mb)
            seq_len_history.append(current_seq)
            
            # 2. Sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token_id.item())
            
            # 3. Prepare next input
            input_ids = next_token_id
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
            
            # 4. Eviction Logic (Adapted from your class)
            # Start eviction check after warmup
            if i >= 5 and past_key_values is not None:
                if current_seq > max_cache_size:
                    tokens_to_evict = current_seq - max_cache_size
                    eviction_events.append(step_count) # Mark this step
                    
                    # Simple FIFO Eviction for UI stability (Attention based is heavy for UI)
                    # Using your logic structure but simplified for speed in UI
                    evict_count = tokens_to_evict
                    
                    if hasattr(past_key_values, 'layers') and past_key_values.layers is not None:
                        new_cache = DynamicCache()
                        new_cache.layers = []
                        for layer in past_key_values.layers:
                            k_layer = layer.key if hasattr(layer, 'key') else None
                            v_layer = layer.value if hasattr(layer, 'value') else None
                            if k_layer is not None and v_layer is not None:
                                new_k = k_layer[:, :, evict_count:, :]
                                new_v = v_layer[:, :, evict_count:, :]
                                layer_type = type(layer)
                                new_layer = layer_type()
                                if hasattr(new_layer, 'key'): new_layer.key = new_k
                                if hasattr(new_layer, 'value'): new_layer.value = new_v
                                new_cache.layers.append(new_layer)
                        if hasattr(new_cache, '_seen_tokens'):
                            new_cache._seen_tokens = current_seq - evict_count
                        past_key_values = new_cache
                    
                    # Update tracking immediately after eviction to show drop
                    cache_mb = get_cache_memory_mb(past_key_values)
                    current_seq = get_cache_seq_len(past_key_values)
                    # Overwrite the last recorded point to show the drop in the graph
                    if cache_history: 
                        cache_history[-1] = cache_mb
                        seq_len_history[-1] = current_seq

            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            step_count += 1
            
            # Yield partial text for streaming effect (optional, keeping simple for now)
            # For this demo, we return full text + metrics to update plot cleanly
            
    elapsed = time.time() - start_time
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text, cache_history, seq_len_history, eviction_events, elapsed, len(generated_ids)

def create_plot(cache_history, seq_len_history, eviction_events, max_cache_size):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Memory
    color = 'tab:blue'
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('Cache Memory (MB)', color=color)
    ax1.plot(cache_history, color=color, label='Active Cache Memory')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot Threshold Line
    # Approximate threshold MB based on max_cache_size (rough estimate for visualization)
    # Since MB depends on model dim, we just draw a line at the max observed before eviction
    if cache_history:
        max_observed = max(cache_history)
        ax1.axhline(y=max_observed * 0.9, color='red', linestyle='--', label='Eviction Threshold Trigger')

    # Mark Evictions
    if eviction_events:
        ax1.scatter(eviction_events, [cache_history[i] if i < len(cache_history) else 0 for i in eviction_events], 
                    color='red', s=50, zorder=5, label='Eviction Event')

    ax1.legend(loc="upper left")
    plt.title(f"KV Cache Eviction Visualization (Max Size: {max_cache_size} tokens)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def chat_interface(message, history, max_cache_size, max_new_tokens):
    if not message:
        return "", history, None, ""
    
    # Format conversation history for the model
    # Qwen2.5 Instruct format
    conversation = ""
    for user_msg, bot_msg in history:
        conversation += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
    
    conversation += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Run Generation
    response, cache_hist, seq_hist, evictions, time_taken, tokens = generate_with_eviction(
        conversation, 
        max_cache_size=int(max_cache_size), 
        max_new_tokens=int(max_new_tokens), 
        temperature=0.7
    )
    
    # Update History
    history = history + [[message, response]]
    
    # Create Plot
    plot = create_plot(cache_hist, seq_hist, evictions, max_cache_size)
    
    # Stats String
    stats = (f"⏱️ Time: {time_taken:.2f}s\n"
             f"📝 Tokens: {tokens}\n"
             f"🗑️ Evictions: {len(evictions)}\n"
             f"💾 Max Cache: {max(cache_hist):.2f} MB" if cache_hist else "")
    
    return "", history, plot, stats

# --- Gradio UI ---

with gr.Blocks(title="LLM Cache Eviction Demo") as demo:
    gr.Markdown("## 🧠 Adaptive KV Cache Eviction Visualizer")
    gr.Markdown("Chat with the model. The graph on the right shows memory usage. **Red dots** indicate when the cache exceeded the limit and tokens were evicted.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            with gr.Row():
                msg = gr.Textbox(show_label=False, placeholder="Type a message...", scale=4)
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.ClearButton([chatbot, msg])
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            cache_slider = gr.Slider(minimum=50, maximum=512, value=150, step=10, label="Max Cache Size (Tokens)")
            token_slider = gr.Slider(minimum=50, maximum=500, value=250, step=10, label="Max New Tokens")
            
            gr.Markdown("### 📊 Live Metrics")
            stats_box = gr.Textbox(label="Generation Stats", lines=4, interactive=False)
            plot_output = gr.Plot(label="Cache Memory History")
            
    # Event Listeners
    send_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot, cache_slider, token_slider],
        outputs=[msg, chatbot, plot_output, stats_box]
    )
    
    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot, cache_slider, token_slider],
        outputs=[msg, chatbot, plot_output, stats_box]
    )

if __name__ == "__main__":
    demo.launch()
