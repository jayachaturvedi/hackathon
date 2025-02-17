# todo, to handle this properly, I should have generated all diaglogs first then send each dialog
import argparse
import json
import torch
from transformers import AutoTokenizer
from llama_models.llama3.reference_impl.generation import Llama  # Keep using Meta's code
from llama_models.llama3.api.datatypes import RawMessage, StopReason
import pandas as pd

from typing import Optional
def read_csv(file_path):
    with open(file_path, 'r') as f:
        #read them into json records
        data_df = pd.read_csv(f, sep=',')
        data_dict = data_df.to_dict(orient="records")

    return data_dict

def main(
        all_prompt_keys,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0,
        top_p: float = 0,
        max_seq_len: int = 8192,
        max_batch_size: int = 8,
        max_gen_len: int = 100
):
    # Load Meta's Llama model
    print(f"Loading model from {ckpt_dir}...")
    # huggingface loads models using AutoModelForCausalLM, and requires the model file to be named 'pytorch_model.bin'
    # the model i downloaded from meta uses 'model.ckpt' as the model file name. therefore, use meta builtin loading function.
    # no need to load the model and  seperately
    # some implementation uses Llama, some uses Llama.build
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print("Total prompts to be processed:", len(all_prompt_keys))

    all_outputs = []
    for count, item in enumerate(all_prompt_keys, start=1):
        print(f"----Processing prompt #{count}: {item}")
        print(item)
        note_id = item["note_ID"]
        print("^^^^note_id:", note_id)
        context = item["text"]
        entity_text = item["ner_text"]
        print("----ner_text", entity_text)
        entity_type = item["ner_type"]
        print("£££ner_type", entity_type)
        
        # Create a prompt for generation
        prompt_symptom = """
            You are a medical expert. I am providing you with medical data from the MIMIC dataset and some symptoms I have extracted.
            Please help me check if the symptoms actually happened to the patient based on the context, as compared to just being mentioned.
            Also if it is a adverse effect(ADR) or not. 
            Make sure you return a valid json that strictly follows the format below, 
            [
                  {
                    "entity": "<entity text>",
                    "entity_type": "<entity label>",
                    "happened": <TRUE/FALSE>,
                    "ADR":<TRUE/FALSE>
                    }
            ]
            """
        prompt_meds = """
                        You are a medical expert. I am providing you with medical data from the MIMIC dataset and some medications I have extracted.
                Please help me check if the medications actually happened to the patient based on the context, as compared to just being mentioned.
                Also if it is a adverse effect(ADR) or not. 
                Make sure you return a valid json that strictly follows the format below, 
                      {
                        "entity": "<entity text>",
                        "entity_type": "<entity label>",
                        "happened": <TRUE/FALSE>,
                        }
                    """
        if entity_type == "symptom":
            prompt_sys = prompt_symptom
        if entity_type == "medication":
            prompt_sys = prompt_meds
        prompt_usr = f"""Here is the context: {context}. The {entity_type} entity: {entity_text}. Give me a valid json object only, strictly following the JSON format. \
                     """

        dialog = [RawMessage(role="system", content=prompt_sys), RawMessage(role="user", content=prompt_usr)]

        result = generator.chat_completion(
            dialog,  # Pass a single message list to keep it independent
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # access the output properly
        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        a_whole = {"note_ID": note_id,  "result": out_message.content}
        all_outputs.append(a_whole)
        print("=========")

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate medical definitions using Llama (Meta checkpoint format).")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Checkpoint directory for Llama model.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument('--temperature', type=float, default=0, help="Temperature for generation.")
    parser.add_argument('--top_p', type=float, default=0, help="Top P for nucleus sampling.")
    parser.add_argument('--max_seq_len', type=int, default=8192, help="Maximum sequence length.")
    parser.add_argument('--max_batch_size', type=int, default=8, help="Maximum batch size.")
    parser.add_argument('--data_f_csv', type=str, required=True, help="Path to the JSON file containing all prompts.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")

    args = parser.parse_args()

    # Load all prompt keys from JSON file
    all_prompt_keys = read_csv(args.data_f_csv)

    # Run main function
    all_outputs = main(
        all_prompt_keys=all_prompt_keys,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size
    )
    with open(args.output_file, 'w') as f:
        json.dump(all_outputs, f, indent=4)
