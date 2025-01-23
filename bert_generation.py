import mindspore
import numpy as np
from mindnlp.transformers import BertTokenizer, BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder
from mindspore import Tensor
from rouge import Rouge
from datasets import load_dataset

# 1. 使用 Hugging Face 加载 BBC 数据集
# 加载数据集
dataset = load_dataset("SetFit/bbc-news")

# 查看测试集包含多少条文本
num_test_samples = len(dataset['test'])
print(f'Test dataset contains {num_test_samples} samples.')

# 2. 假设我们只关心新闻标题和内容，取出输入（标题）和目标（内容）
input_texts = dataset['test']['text'][:3]  # 获取测试集前三条新闻文本
target_texts = dataset['test']['text'][:3]  # 目标文本应该是新闻内容

# 3. 设置模型配置
vocab_size = 30522  # 词汇表大小
hidden_size = 768  # 隐藏层大小
num_hidden_layers = 12  # 隐藏层数
num_attention_heads = 12  # 注意力头数
intermediate_size = 3072  # 中间层大小
hidden_act = "gelu"  # 激活函数
hidden_dropout_prob = 0.1  # 隐藏层的dropout比例
attention_probs_dropout_prob = 0.1  # 注意力层的dropout比例
max_position_embeddings = 512  # 最大位置编码数
initializer_range = 0.02  # 初始化范围
layer_norm_eps = 1e-12  # 层归一化的epsilon值
feed_forward_size = 3072  # 前馈层的大小
classifier_dropout = 0.1  # 分类器的dropout比例

# 加载BertGeneration配置
config = BertGenerationConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    hidden_act=hidden_act,
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    max_position_embeddings=max_position_embeddings,
    initializer_range=initializer_range,
    layer_norm_eps=layer_norm_eps,
    feed_forward_size=feed_forward_size,
    is_decoder=False,
    use_cache=True,
    add_cross_attention=False,
    classifier_dropout=classifier_dropout,
)

# 4. 加载预训练模型
model_encoder = BertGenerationEncoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
model_decoder = BertGenerationDecoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")

# 5. 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 6. 对测试集中的前三条数据进行预处理并生成文本
rouge = Rouge()
for i in range(3):
    print(f"Processing test sample {i + 1}:")

    # 对每一条文本进行tokenization
    input_ids = tokenizer(input_texts[i], return_tensors='np', padding=True, truncation=True, max_length=512)[
        'input_ids']

    # 确保 input_ids 为 np.int64 类型
    input_ids = np.asarray(input_ids, dtype=np.int64)

    # ** 将 input_ids 转换为 Tensor 类型 **，使用 int64
    input_ids = Tensor(input_ids, dtype=mindspore.int64)

    # 获取 Encoder 的输出
    encoder_output = model_encoder(Tensor(input_ids))

    # Decoder的输入需要包含Encoder的输出（隐状态）
    encoder_hidden_states = encoder_output.last_hidden_state

    # 使用 np.ones_like 创建一个相同形状的全 1 数组，确保数据类型为 np.int64
    encoder_attention_mask = np.ones_like(input_ids.asnumpy(), dtype=np.int64)

    # ** 将 encoder_attention_mask 转换为 Tensor 类型 **，使用 int64
    encoder_attention_mask = Tensor(encoder_attention_mask, dtype=mindspore.int64)

    # 生成文本
    generated_ids = model_decoder.generate(
        input_ids=input_ids,  # 输入token ID
        encoder_hidden_states=encoder_hidden_states,  # Encoder输出的隐状态
        encoder_attention_mask=encoder_attention_mask,  # Encoder的attention mask
        max_new_tokens=20,  # 仅限制生成部分的最大长度
    )

    # 解码生成的token ID为文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    # 评估生成文本（使用 ROUGE 指标）
    scores = rouge.get_scores(generated_text, target_texts[i])  # 使用目标文本进行评估
    print(f"ROUGE Scores for sample {i + 1}: {scores}")
    print("-" * 50)  # 分隔每一条数据的结果
