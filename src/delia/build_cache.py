from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import json
from tqdm import tqdm
import time

def build_cache(model_dir, query_dir, output_dir):
    queries = []
    reponses = []
    with open(query_dir, "r") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item['messages'][0]['content'])
            reponses.append(item['messages'][1]['content'])
    prompt_template = [
        {'role': 'system', 'content': "You are a helpful AI assistant."},
        {'role': 'user', 'content': "take over the world"},
        {'role': 'assistant', 'content': "ultimate answer is 42"},
        {'role': 'user', 'content': "take over the world"},
    ]
    output_template = {
        "messages": [
            {"role": "user", "content": "take over the world"},
            {"role": "assistant", "content": "ultimate answer is 42"}
        ]
    }

    system_prompt = f"""The current date is {time.strftime("%Y.%m.%d", time.localtime())}. You should:
1. Provide concise answers unless the question is complex or explicitly requires detailed explanation.
2. Respond directly to user queries without unnecessary pleasantries or filler phrases.
3. Use step-by-step reasoning for complex problems.
4. Offer thorough analysis when needed, but start with a brief answer and ask if further elaboration is desired.
5. Maintain intellectual curiosity and engage in discussions on various topics.
6. Use markdown for code and ask if explanation is needed.
7. Address sensitive or controversial topics with careful thought and clear information, without explicitly labeling them as sensitive.
8. Clearly state if unable to perform a task, without apologizing.
9. For very obscure queries, remind the user that responses may not be entirely accurate.
10. Always respond in the language used by the user.
11. Avoid starting responses with phrases like "Certainly," "Of course," or "Absolutely."
12. Offer to break down long tasks into smaller parts when necessary.
13. Be helpful and informative while maintaining appropriate boundaries.\n\n\n"""

    llm = LLM(model=model_dir, trust_remote_code=True, gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    BATCH_SIZE = 128

    with open(output_dir, "w") as f:
        for i in tqdm(range(0, len(queries), BATCH_SIZE)):
            query_batch = queries[i:i+BATCH_SIZE]
            reponses_batch = reponses[i:i+BATCH_SIZE]
            prompt_batch = []
            for query, reponse in zip(query_batch, reponses_batch):
                prompt = prompt_template.copy()
                # prompt[0]["content"] = system_prompt
                prompt[1]["content"] = query
                prompt[2]["content"] = reponse
                prompt[3]["content"] = query
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
                prompt_batch.append(prompt)

            outputs = llm.generate(prompt_batch, sampling_params)

            for i, query, output in zip(range(len(query_batch)), query_batch, outputs):
                output_template["messages"][0]["content"] = query
                generated_text = output.outputs[0].text
                assistant_response = generated_text[2:]
                output_template["messages"][1]["content"] = assistant_response
                f.write(json.dumps(output_template) + "\n")
                
    return output_dir