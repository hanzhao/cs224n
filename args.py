import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='indomain_train/squad,indomain_train/nat_questions,indomain_train/newsqa')
    parser.add_argument('--eval-datasets', type=str, default='indomain_val/squad,indomain_val/nat_questions,indomain_val/newsqa')
    parser.add_argument('--augment-datasets', type=str, default='oodomain_train/race,oodomain_train/relation_extraction,oodomain_train/duorc')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default='')
    parser.add_argument('--data-dir', type=str, default='datasets')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--eval-after-epoch', action='store_true')
    parser.add_argument('--augment-methods', type=str, default='')
    # in-context
    parser.add_argument('--in-context', action='store_true')
    parser.add_argument('--double-demo', action='store_true')
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--num-sample', type=int, default=16)
    parser.add_argument('--mapping', type=str, default="{str(i):str(i) for i in range(512)}")
    parser.add_argument('--prompt', action='store_true')
    parser.add_argument('--template-list', type=str, default=None)
    parser.add_argument('--template', type=str, default="*cls**sent_0*_?_answer_starts_at*mask*_and_ends_at*mask**sep**sent_1**sep+*")
    parser.add_argument('--use-demo', action='store_true')
    parser.add_argument('--demo-filter', action='store_true')
    args = parser.parse_args()
    return args
