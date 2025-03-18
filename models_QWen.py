from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen:
    def __init__(self, model_name, load_checkpoint=False, checkpoint_path=None):
        self.runtime_env = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if load_checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype="auto", device_map="auto")
            self.model.to(self.runtime_env)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            self.model.to(self.runtime_env)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, p):
        conv_data = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant to solve math problems."},
            {"role": "user", "content": p}
        ]
        text_input = self.tokenizer.apply_chat_template(conv_data, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**encoded, max_new_tokens=512, temperature=0.6, top_p=0.95)
        filtered = [ids[len(inp):] for inp, ids in zip(encoded.input_ids, output)]
        result = self.tokenizer.batch_decode(filtered, skip_special_tokens=True)[0]
        return result

    def batch_inference(self, questions):
        data_list = [[{"role": "user", "content": q}] for q in questions]
        text_batch = self.tokenizer.apply_chat_template(data_list, tokenize=False, add_generation_prompt=True)
        batch_encoded = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)
        batch_output = self.model.generate(**batch_encoded, max_new_tokens=512, temperature=0.6, top_p=0.95)
        final_ids = batch_output[:, batch_encoded.input_ids.shape[1]:]
        results = self.tokenizer.batch_decode(final_ids, skip_special_tokens=True)
        return results

if __name__ == "__main__":
    model = Qwen("Qwen2.5-0.5B-Instruct", load_checkpoint=False, checkpoint_path="Qwen2.5-0.5B-Instruct-GRPO/checkpoint-6000")
    question = "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
    answer = model.inference(question)
    print(answer)
