import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from MRT.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from MRT.conversation import conv_templates, SeparatorStyle
from MRT.model.builder_MRTTrained import load_pretrained_model
from MRT.utils import disable_torch_init
from MRT.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return data_loader


def load_rt_modules(model, result_dir):
    """Load RT modules state_dict into the model with proper dtype handling"""
    rt_modules_path = os.path.join(result_dir, "rt_modules.bin")
    
    if os.path.exists(rt_modules_path):
        rt_modules = torch.load(rt_modules_path, map_location='cpu')
        
        # Load each RT module's state_dict
        for module_name, state_dict in rt_modules.items():
            # Find the corresponding module in the model
            module = model
            for attr in module_name.split('.'):
                module = getattr(module, attr)
            
            # Filter out cache entries and handle dtypes appropriately
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if '_weight_t_cache' not in key:
                    # Keep rotate_layer weights in float32, convert others to bfloat16
                    if 'rotate_layer.weight' in key:
                        filtered_state_dict[key] = value.to(torch.float32)
                    else:
                        filtered_state_dict[key] = value.to(torch.bfloat16)
            
            # Load the filtered state_dict into the module
            module.load_state_dict(filtered_state_dict, strict=False)
            
            # Set module dtypes appropriately
            if hasattr(module, 'rotate_layer'):
                # Keep rotate_layer weight in float32
                module.rotate_layer.weight.data = module.rotate_layer.weight.data.to(torch.float32)
                # Invalidate any cached transposed weights
                module.rotate_layer._weight_t_cache_valid = False
                module.rotate_layer._weight_t_cache = None
            
            if hasattr(module, 'learned_source'):
                # Convert learned_source to bfloat16
                module.learned_source.to(torch.bfloat16)
            
            # Update the module's dtype attribute if it exists
            if hasattr(module, 'dtype'):
                module.dtype = torch.bfloat16
        
        return True
    else:
        return False


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        LM_rank=args.RT_rank_llm, 
        V_rank=args.RT_rank_vision
    )
    
    # Load RT modules if result_dir is provided
    if args.result_dir and os.path.exists(args.result_dir):
        print(f"Loading saved RT weights from {args.result_dir}")
        rt_loaded = load_rt_modules(model, args.result_dir)
        if not rt_loaded:
            print("Warning: No RT modules loaded!")
    
    # Load mm_projector if exists
    if args.result_dir:
        mm_projector_path = os.path.join(args.result_dir, "mm_projector.bin")
        if os.path.exists(mm_projector_path):
            print(f"Loading mm_projector from {mm_projector_path}")
            mm_projector_weights = torch.load(mm_projector_path, map_location='cpu')
            mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
            print("Successfully loaded mm_projector")
        
        # Load lm_head if exists
        lm_head_path = os.path.join(args.result_dir, "lm_head.bin")
        if os.path.exists(lm_head_path):
            print(f"Loading lm_head from {lm_head_path}")
            lm_head_weights = torch.load(lm_head_path, map_location='cpu')
            lm_head_weights = {k: v.to(torch.bfloat16) for k, v in lm_head_weights.items()}
            model.load_state_dict(lm_head_weights, strict=False)
            print("Successfully loaded lm_head")


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(args.answers_file, exist_ok=True)
    ans_dict = {
        "existence": [], "count": [], "position": [], "color": [], "posters": [], "celebrity": [], "scene": [],
        "landmark": [], "artwork": [],
        "OCR": [], "commonsense_reasoning": [], "numerical_calculation": [], "text_translation": [],
        "code_reasoning": []
    }
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    i=1
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        category = line["category"]
        cur_prompt = line["text"]
        question_id = line['question_id']
        target = line["target"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Model: {outputs}")
        print(f"Target: {target}")
        print("--------------------------")
        single_output = json.dumps({"question_id": question_id,
                                    "prompt": cur_prompt,
                                    "target": target,
                                    "predict": outputs}) + "\n"
        ans_dict[category].append(single_output)

    for categ in ans_dict.keys():
        with open(os.path.join(args.answers_file, categ + ".jsonl"), "w") as f:
            for line in ans_dict[categ]:
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--RT_rank_llm", type=int, default=4)
    parser.add_argument("--RT_rank_vision", type=int, default=6)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)