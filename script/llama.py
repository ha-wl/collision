from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.9, max_tokens=512, stop="<|eot_id|>"
)


message = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": "二つの物体が衝突するという状況を考えます。「立方体が円柱と衝突する。立方体の質量が小さい。円柱の質量が大きい。」という条件のとき、衝突時の状況は「円柱は勢いよく立方体に衝突して、立方体は弾き飛ばされる」のように簡潔に描写されるとします。では「立方体が円柱と衝突する。立方体の質量がかなり小さい。円柱の質量がかなり大きい。」という条件のときの衝突時の描写はどのようになるでしょうか。",
    },
]
prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)

output = llm.generate(prompt, sampling_params)

print(output[0].outputs[0].text)
