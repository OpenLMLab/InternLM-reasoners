import warnings
from typing import Tuple, Union, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .. import LanguageModel, GenerateOutput

class InternLMModel(LanguageModel):
    def __init__(self, path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
           path, trust_remote_code=True
        ).cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True
        )

    def generate(self,
                 inputs: List[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, List[str], List[int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 **kwargs) -> GenerateOutput:
        tokenized = self.tokenizer(inputs, max_length=max_length,
                                   padding=True, truncation=True,
                                   return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        output = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, eos_token_id=eos_token_id, top_k=top_k,
            top_p=top_p, num_return_sequences=num_return_sequences,
            return_dict_in_generate=True, output_scores=True
        )
        decoded = []
        prompt_len = input_ids.shape[1]
        sequences = output.sequences
        if hide_input:
            sequences = sequences[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # scores: new_length of batch_size, vocab_size
        seq_prob = []
        for i, score in enumerate(output.scores):
            score = torch.softmax(score, dim=-1)
            pos = prompt_len + i
            token = output.sequences[:, pos]
            seq_prob.append(score[list(range(score.shape[0])), token])

        log_prob = None
        if output_log_probs:
            seq_probs = torch.stack(seq_prob, dim=1)
            # batch_size, seq_len
            log_prob = torch.log(seq_probs)

        return GenerateOutput(decoded, log_prob)

    def get_next_token_logits(self,
                              prompt: Union[str, List[str]],
                              candidates: Union[List[str], List[List[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs):
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer(cand, add_special_tokens=False).input_ids
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[0])
        input_ids = self.tokenizer(prompt, padding=True, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            all_logits = self.model(input_ids).logits[:, -1, :]
        
        logits = []
        if postprocess == "log_softmax":
            all_logits = torch.log_softmax(all_logits, dim=-1)
        elif postprocess == "softmax":
            all_logits = torch.softmax(all_logits, dim=-1)
        for case_logits, cand in zip(all_logits, cand_tokens):
            print(cand, torch.argmax(case_logits, dim=-1), self.tokenizer.decode(torch.argmax(case_logits, dim=-1).cpu().tolist()))
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    def get_loglikelihood(self,
                          prefix: str,
                          contents: List[str]):
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        bsz = len(contents)

        prefix_tokens = self.tokenizer(prefix).input_ids
        prompts_tokens = self.tokenizer(contents, padding=True, return_tensors="pt").input_ids.cuda()
        for prompt_tokens in prompts_tokens:
            assert prompt_tokens[: len(prefix_tokens)].tolist() == prefix_tokens

        with torch.no_grad():
            logits = self.model(prompts_tokens).logits
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), prompts_tokens.shape[1]):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if prompts_tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, prompts_tokens[j, i]])

        return acc_probs.cpu().numpy()
