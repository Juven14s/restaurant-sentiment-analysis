"""CPU study: no neural training, a frozen probe, and explicit format ablation."""

import argparse
import json
from pathlib import Path

from restaurant_sentiment.analysis import diagnose, format_ablation
from restaurant_sentiment.baselines import run_baseline
from restaurant_sentiment.benchmark import category_scores, freeze_challenge, freeze_validation_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/light-study-v1'))
    parser.add_argument('--benchmark-dir', type=Path, default=Path('benchmarks/fr_aspect_challenge_v1'))
    parser.add_argument('--run-llm', action='store_true', help='Download/load the small model and run inference only')
    parser.add_argument('--model-name', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--model-revision', default='c1899de289a04d12100db370d81485cdf75e47ca')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error('--cpu-threads must be positive')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    freeze_challenge(args.benchmark_dir/'cases.json',args.benchmark_dir)
    probe_dir = root/'validation_probe'
    freeze_validation_probe(args.data_dir/'validation.jsonl',probe_dir,size=40,seed=42)
    for method in ('majority','tfidf'):
        run_baseline(args.data_dir,root/'validation'/method,method,evaluation_split='validation')
        run_baseline(args.data_dir,root/'probe'/method,method,evaluation_path=probe_dir/'probe.jsonl')
        run_baseline(args.data_dir,root/'challenge'/method,method,evaluation_path=args.benchmark_dir/'challenge.jsonl')
    analysis_dir = root/'validation'/'analysis'
    if not (analysis_dir/'error_review.csv').exists():
        diagnose(args.data_dir/'validation.jsonl',root/'validation'/'tfidf'/'predictions.jsonl',analysis_dir)
    else:
        print('Existing review worksheet preserved:',analysis_dir/'error_review.csv')
    methods = ['majority','tfidf']
    if args.run_llm:
        import torch
        from restaurant_sentiment.training import predict_model

        torch.set_num_threads(args.cpu_threads)
        for name,reference in (('probe',probe_dir/'probe.jsonl'),
                               ('challenge',args.benchmark_dir/'challenge.jsonl')):
            predictions = root/name/'qwen'/'predictions.jsonl'
            predict_model(reference,predictions,args.model_name,'llm',device='cpu',batch_size=1,
                          max_length=512,max_new_tokens=96,model_revision=args.model_revision)
            format_ablation(reference,predictions,root/name/'qwen'/'ablation')
        methods.append('qwen')
    slices = {}
    for method in methods:
        predictions = root/'challenge'/method/'predictions.jsonl'
        slices[method] = category_scores(args.benchmark_dir/'challenge.jsonl',predictions,
                                        args.benchmark_dir/'annotations.json')
    if args.run_llm:
        slices['qwen_single_fence'] = category_scores(args.benchmark_dir/'challenge.jsonl',
            root/'challenge'/'qwen'/'ablation'/'predictions_single_fence.jsonl',args.benchmark_dir/'annotations.json')
    (root/'challenge_categories.json').write_text(json.dumps(slices,ensure_ascii=False,indent=2)+'\n')
    print('Study artifacts:',root)
    print('Review the private validation worksheet; model-written annotations require independent checking.')


if __name__ == '__main__':
    main()
