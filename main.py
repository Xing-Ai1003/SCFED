import os
from sklearn.model_selection import KFold
import argparse
from utils import *
from models import *
from dgl.dataloading import GraphDataLoader
import json


# arguments for cff explainer
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--is_train', type=bool, default=False)
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--in_dim', type=int, default=74)
parser.add_argument('--h_dim', type=int, default=8)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--num_rels', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr_outer', type=float, default=1e-3)
parser.add_argument('--lr_inner', type=float, default=1e-3)
parser.add_argument('--epochs_outer', type=int, default=50)
parser.add_argument('--epochs_inner', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_fold', type=int, default=10)
parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5,
                    help="threshold to convert relaxed adj matrix to binary")
args = parser.parse_args()


# Initialize parameters
opcodes = [
    "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ",
    "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT",
    "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP",
    "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL",
    "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER",
    "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE",
    "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}
num_opcodes = len(opcode_to_feature)
kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=42)
fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# dataset_path = 'data/creation_1346_shuffled.pkl'
dataset_path = 'data/creation_1346_shuffled.pkl'
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
for g in dataset.graphs:
    g.edata['eweight'] = torch.ones_like(g.edata['etype'])

fold_precision, fold_recall, fold_f1 = [], [], []
for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Starting fold {fold + 1}/{args.num_fold}")
    train_set, val_set, test_set, train_tm, val_tm, test_tm = \
        load_data(dataset, train_val_idx, test_idx, process_type=[])#'block',#'few_samples',#'flip'

    # Create dataloaders
    train_dataloader = GraphDataLoader(train_set, batch_size=args.batch_size, drop_last=False)
    val_dataloader = GraphDataLoader(val_set, batch_size=args.batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(test_set, batch_size=args.batch_size, drop_last=False)

    # Create model
    semodel = SelfExplainableRGCN(args, device)

    # training
    if args.is_train:
        semodel.fit(train_dataloader, val_dataloader, test_dataloader, fold)

    # test
    best_epoch = 50
    best_model_state = torch.load(f'./trained_model/best_{fold + 1}_{best_epoch}.pt')
    semodel.encoder.load_state_dict(best_model_state['encoder'])
    semodel.mlp.load_state_dict(best_model_state['mlp'])
    precision, recall, f1, precision_sub, recall_sub, f1_sub = semodel.test(test_dataloader)
    fold_precision.append(precision)
    fold_recall.append(recall)
    fold_f1.append(f1)
print(f"10-Fold Results: Precision: {np.mean(np.array(fold_precision)):.2f}, "
      f"Recall: {np.mean(np.array(fold_recall)):.2f}, "
      f"F1_binary: {np.mean(np.array(fold_f1)):.2f}\n")