import os
from pathlib import Path
from util import *
from eval import *
from dataset import get_dataset
from prompts import PromptFactory
from model import get_model
from tqdm import tqdm
from pprint import pprint


def launch():
    args = parse_args()
    pprint(args)

    # output
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)

    # resume
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if 'data' in processed:
            processed = processed['data']

    # get input
    quids_to_exclude = set(list(processed.keys()))
    dataset = get_dataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run)

    # configure prompt
    prompter = PromptFactory().get(args.prompt_type)

    # get model
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)

    # answer
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        clip_length = int(1/args.fps) if args.fps < 1 else 1/args.fps
        clip_length *= args.caption_every
        few_shot_examples = build_fewshot_examples(args.fewshot_example_path, args.data_path)
        prompt = prompter.fill(**item, fps=args.fps, clip_length=clip_length, caption_every=args.caption_every, num_words=args.num_words_in_sum, examplars=few_shot_examples)
        pred, info = model.forward(prompter.head, prompt)
        ukey_name = dataset.ukey
        ukey = item[ukey_name]
        processed[ukey] = item
        processed[ukey]['pred'] = pred
        processed[ukey]['prompt_template'] = prompter.get_template_str()
        processed[ukey]['response'] = info['response']
        if args.save_info:
            processed[ukey]['info'] = {k: v for k, v in info.items() if k != 'response'}
        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)

    # incorporate with backup prediction
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        if 'data' in backup:
            backup = backup['data']
        for uid in processed:
            if processed[uid]['pred'] == -1:
                processed[uid]['pred'] = backup[uid]['pred']

    # if args.dataset == 'videomme':
    #     eval_output_path = os.path.join(args.output_base_path, f'{args.output_filename}_temp.json')
    #     processed_videomme = restore_videomme(processed, args.anno_path)
    #     print(f'number of videos: {len(processed_videomme)}')
    #     save_json(processed_videomme, eval_output_path)

    save_json(processed, output_path)

    # if eval
    if not args.disable_eval:
        if args.task == 'qa':
            if args.dataset == 'egoschema':
                processed = eval_qa_egoschema(processed)
                save_json(processed, output_path)
            elif args.dataset in ['nextqa', 'intentqa', 'nextgqa']:
                processed = eval_qa_nextqa(args.anno_path, processed)
                save_json(processed, output_path)
            elif args.dataset == 'videomme':
                processed = eval_qa_videomme(processed)
                save_json(processed, output_path)
                # eval_videomme(eval_output_path, args.videomme_type)
        elif args.task == 'gqa':
            if args.dataset == 'nextgqa':
                pred_qa_path = args.nextgqa_pred_qa_path if len(args.nextgqa_pred_qa_path) > 0 else None
                processed = eval_gqa(args.nextgqa_gt_ground_path, processed, pred_qa_path=pred_qa_path)
            save_json(processed, output_path)
        elif args.task == 'sum':
            processed, sum_data = eval_sum(processed)
            save_json(sum_data, f'{Path(output_path).parent / Path(output_path).stem}_data.json')



if __name__ == '__main__':
    launch()
    