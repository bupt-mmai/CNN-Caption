import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Image Paragraph Caption Model')

    # Data input settings

    parser.add_argument('--para2vec_path', type=str, default='data/img2para_vec.pkl', \
                        help='Path of image to paragraph index and stop flag  dict: {str: [word index, stop flag]}')
    parser.add_argument('--densecap_path', type=str, default='data/img2dense_vec.pkl', \
                        help='Path of dense caption vector dict {str: caption index}')
    parser.add_argument('--img2para_path', type=str, default='data/img2paragraph', \
                        help='Path of image to paragraph  dict:{int: [sent]}')
    parser.add_argument('--train_img_path', type=str, default='data/train_split.json', \
                        help='Path of train set image name')
    parser.add_argument('--train_feats_path', type=str, default='data/train_feats.h5', \
                        help='Path of train set image features')
    parser.add_argument('--test_img_path', type=str, default='data/test_split.json', \
                        help='Path of test set image name')
    parser.add_argument('--test_feats_path', type=str, default='data/test_feats.h5', \
                        help='Path of test set image features')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', \
                        help='Path of vocab')

    # Data save settings
    parser.add_argument('--model_dir', type=str, default='model_files', \
                        help='path of model files')
    parser.add_argument('--eval_dir', type=str, default='eval', \
                        help='path of evaluation files')
    parser.add_argument('--log_dir', type=str, default='log', \
                        help='path of logging files')

    # Model parameters settings
    parser.add_argument('--s_max', type=int, default=6,
                        help='maximum number of sentences')
    parser.add_argument('--w_max', type=int, default=30,
                        help='maximum number of words in a sentence')
    parser.add_argument('--num_boxes', type=int, default=50,
                        help='number of boxes in an image')
    parser.add_argument('--s_kernel_size', type=int, default=3,
                        help='kernel size of sentence CNN')
    parser.add_argument('--w_kernel_size', type=int, default=10,
                        help='kernel size of word CNN')
    parser.add_argument('--s_num_layers', type=int, default=1,
                        help='number of Sentence CNN layers')
    parser.add_argument('--w_num_layers', type=int, default=1,
                        help='number of Word CNN layers')
    parser.add_argument('--feat_size', type=int, default=4096,
                        help='size of box feature')
    parser.add_argument('--pad_idx', type=int, default=2,
                        help='index of <pad> token')
    parser.add_argument('--emb_size', type=int, default=1024,
                        help='size of embedding vector')
    parser.add_argument('--proj_size', type=int, default=1024,
                        help='size of project box feature')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=850,
                        help='eval batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of threads when loading data')
    parser.add_argument('--max_epoch', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')

    parser.add_argument('--beam', dest='beam', action='store_true')
    parser.add_argument('--sample', dest='beam', action='store_false')
    parser.set_defaults(beam=True)

    parser.add_argument('--beam_size', type=int, default=2,
                        help='beam size')
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--sent_cost_lambda', type=float, default=5,
                        help='sent cost lambda')

    parser.add_argument('--cuda_id', type=int, default=0,
                        help='cuda id')
    parser.add_argument('--model_name', type=str, default='cnn',
                        help='model files name')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch')
    parser.add_argument('--eval_after', type=int, default=5,
                        help='start evaluation after this epoch')

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)

    parser.add_argument('--gen', dest='gen', action='store_true')
    parser.add_argument('--no-gen', dest='gen', action='store_false')
    parser.set_defaults(gen=True)

    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)

    parser.add_argument('--att', dest='att', action='store_true')
    parser.add_argument('--no-att', dest='att', action='store_false')
    parser.set_defaults(att=True)

    parser.add_argument('--gen_list', nargs='+', help='list of images which need generate', required=True)

    args = parser.parse_args()
    assert args.optim in ['adam', 'sgd', 'rmsprop', 'adagrad'], "optimizer doesn't exist"
    assert args.cuda_id < 4 and args.cuda_id >= 0, "gpu id should be between 0 and 4"
    return args
