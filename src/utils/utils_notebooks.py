import os
import sys
import logging
from collections.abc import Iterator
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
from types import SimpleNamespace
import tqdm
from matplotlib import pyplot as plt
import sklearn.metrics as skm
import sklearn.utils.random as skr
import scipy.stats as sps
from optuna.trial import TrialState
import torch
import torch.nn as nn
from torchinfo import summary
from .dataloader import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn, dynamic_collate_fn
from .utils_cellpainting import save_checkpoint, load_checkpoint
logger = logging.getLogger(__name__)



def run_classification_inference(model, checkpoint_file, epoch, data_loader, data_type, device, report = True, plot = True):
    checkpoint_file = checkpoint_file.format(ep=epoch)+'.pt'
    if not os.path.isfile(checkpoint_file):
        print(f" File: {checkpoint_file} not found!! Continue....")
    else:
        ## load_checkpoint(model, optimizer = None, scheduler = None, metrics = None, filename = None, ckpt_path = '', device = None, dryrun = False, verbose=False):
        model, _, _, _ = load_checkpoint(model, optimizer = None, scheduler = None, metrics = None, filename = checkpoint_file, verbose = False)
        _ = model.eval();
        mdl_outputs = run_model_on_test_data(model, data_loader[data_type], device, title = data_type,  verbose = False)
        if report:
            mo = compute_classification_metrics(mdl_outputs, epochs = epoch, title = data_type, verbose = True)
        if plot:
            plot_cls_metrics(mdl_outputs.y_true, mdl_outputs.y_prob, mdl_outputs.y_pred, title = data_type, epochs = epoch )
            plt.show()

# def run_2model_classification_inference(model1, checkpoint_file1, data_loader1, 
#                                          model2, checkpoint_file2, data_loader2, 
#                                          epoch, data_type, device, report = True, plot = True):
#     ## load_checkpoint(model, optimizer = None, scheduler = None, metrics = None, filename = None, ckpt_path = '', device = None, dryrun = False, verbose=False):
#     checkpoint_file1 = checkpoint_file1.format(ep=epoch)
#     model1, _, _, _ = load_checkpoint(model1, filename = checkpoint_file1, verbose = False)
#     _ = model1.eval();
#     mdl_outputs1 = run_model_on_test_data(model1, data_loader1[data_type], device, title = data_type,  verbose = False)

#     checkpoint_file2 = checkpoint_file2.format(ep=epoch)
#     model2, _, _, _ = load_checkpoint(model2, filename = checkpoint_file2, verbose = False)
#     _ = model2.eval();
#     mdl_outputs2 = run_model_on_test_data(model2, data_loader2[data_type], device, title = data_type,  verbose = False)

#     if report:
#         mo1 = compute_classification_metrics(mdl_outputs1, epochs = epoch, title = data_type, verbose = True)
#         mo2 = compute_classification_metrics(mdl_outputs2, epochs = epoch, title = data_type, verbose = True)
#     if plot:
#         plot_cls_metrics(mdl_outputs1.y_true, mdl_outputs1.y_prob, mdl_outputs1.y_pred, title = data_type, epochs = epoch )
#         plot_cls_metrics(mdl_outputs2.y_true, mdl_outputs2.y_prob, mdl_outputs2.y_pred, title = data_type, epochs = epoch )
#         plot_2model_cls_metrics(mdl_outputs1, mdl_outputs2, title = data_type, epochs = epoch )
#         plt.show()


def run_N_model_classification_inference(models, checkpoint_files, data_loaders, names, epoch = -1, 
                                         data_type = None, size = 8, title = None, device = None, report = True, plot = True):
    model_outputs = []
    data_type = 'test' if data_type is None else data_type
    for mdl, ckpt_file, data_ldr in zip(models, checkpoint_files, data_loaders):
        _ckpt_file = ckpt_file.format(ep=epoch)
        _mdl, _, _, _ = load_checkpoint(mdl, filename = _ckpt_file, verbose = False)
        _ = _mdl.eval();
        _output = run_model_on_test_data(_mdl, data_ldr[data_type], device, title = data_type,  verbose = False)
        # print(type(_output), vars(_output).keys())
        model_outputs.append(_output )

    if report:
        for mdl_output in model_outputs:
            mo = compute_classification_metrics(mdl_output, epochs = epoch, title = data_type, verbose = True)

    if plot:
        for mdl_output, name in zip(model_outputs, names):
            plot_cls_metrics(mdl_output.y_true, mdl_output.y_prob, mdl_output.y_pred, title = title, size = size, epochs = epoch )
    plot_N_model_cls_metrics(model_outputs, names, title = title, size = size, epochs = epoch )
    plt.show()


def compute_classification_metrics(mo, epochs = -1, title = '', verbose = False):
    """
    mo : model outpust from 'run_model_on_test_data'
    """
    cm = SimpleNamespace()
    msg_sfx = f"- epoch:{epochs} " if epochs != -1 else ""
    cm.accuracy = skm.accuracy_score(mo.y_true, mo.y_pred) * 100.0
    cm.roc_auc  = skm.roc_auc_score(mo.y_true, mo.y_logits)
    cm.avg_prec = skm.average_precision_score(mo.y_true, mo.y_pred)
    cm.precision, cm.recall, cm.f1, _ = skm.precision_recall_fscore_support(mo.y_true, mo.y_pred, average='binary', zero_division=0)
    cm.cls_report = skm.classification_report(mo.y_true, mo.y_pred)
    cm.mcc = skm.matthews_corrcoef(mo.y_true, mo.y_pred)
    (mo.y_true == mo.y_pred).sum()
    # cm.test_accuracy = binary_accuracy(y_true=mo.labels, y_prob=mo.logits)
    # cm.test_f1 = binary_f1_score(y_true=mo.labels, y_prob=mo.logits)
    if verbose:
        print(f"{epochs} {title.capitalize():5s} Acc : {cm.accuracy:.2f} %    ROC Auc: {cm.roc_auc:.4f}    PR Auc:{cm.avg_prec:.4f}    F1: {cm.f1:.4f}"
              f"    Prec: {cm.precision:.4f}  Recall: {cm.recall:.4f}  MCC:  {cm.mcc:.4f} ")
        # print(f"\n {title.capitalize()} Label counts:  True: {np.bincount(mo.y_true.squeeze())} \t Pred: {np.bincount(mo.y_pred.squeeze())} ", flush = True)
        # print(f"\n {title.capitalize()} Data Classification Report: {msg_sfx}\n")
        # print(cm.cls_report)
        (mo.y_true == mo.y_pred).sum()
    return cm


def display_cellpainting_batch(batch_id, batch):
    # data, labels, plates, compounds, cmphash, other, labels_2
    features, label, well_ids, compound_id, cmphash, tpsa, label_2 = batch
    # label_2 = np.zeros_like(label)
    print("-"*135)
    print(f"  Batch Id: {batch_id}   {type(batch)}  Rows returned {len(batch[0])} features: {features.shape}  ")
    print(f"+-----+------------------------------------------+----------------+--------------------------+------------------------------+-----+-----+--------------------------------------------------------+")
    print(f"| idx |   batch[2]                               |    batch[3]    |      batch[2]            |          batch[5]            | [1] | [1] |     batch[0]                                           | ") 
    print(f"|     | SRCE      BATCH     PLATE     WELL       |   COMPOUND_ID  |       CMPHASH / BIN      |  TPSA / Ln(TPSA) / Log(TPSA) | LBL |LBL2 |     FEATURES                                           | ")
    print(f"+-----+------------------------------------------+----------------+--------------------------+------------------------------+-----+-----+--------------------------------------------------------+")
         ###    0 | source_11 Batch2    EC000046  K04      | JCP2022_009278 |  7406361908543180200 -  8  |   0   |   62.78000    4.13964   1.79782 | [-0.4377299 -0.4474466  1.1898487  0.2051901]
         # "  1 | source_10    | JCP2022_006020 | -9223347314827979542 |   10 |  0 | tensor([-0.6346, -0.6232, -1.6046])"

    for i in range(len(label)):
        print(f"| {i:3d} | {well_ids[i,0][:9]:9s} {well_ids[i,1][:12]:12s}  {well_ids[i,2][:10]:10s}  {well_ids[i,3]:4s} |"\
              f" {compound_id[i]:14s} | {cmphash[i,0]:20d}  {cmphash[i,1]:2d} |"\
              f" {tpsa[i,0]:7.3f}  {tpsa[i,1]:8.5f}  {tpsa[i,2]:8.5f}  |"
              f" {int(label[i]):2d}  | {int(label_2[i]):2d}  |"\
              f" {features[i,:4].detach().cpu().numpy()}")
        # print(f"| {i:3d} | {batch[2][i,0]:9s} {batch[2][i,1][:9]:9s} {str(batch[2][i,2])[:9]:9s} {batch[2][i,3]:>4s}       "\
        #       f"|{batch[3][i]:12s} | {batch[4][i,0]:20d} - {batch[4][i,1]:2d}  "\
        #       f"{batch[5][i,0]:11.5f}   {batch[5][i,1]:8.5f}  {batch[5][i,2]:8.5f} "\
        #       f"|  {int(batch[1][i]):1d}  | {batch[0][i,:4].detach().cpu().numpy()}")


def display_pharmacophore_batch(batch_id, batch):
    compound_id, cmphash, label = batch
    # label_2 = np.zeros_like(label)
    print("-"*135)
    print(f"  Batch Id: {batch_id} {type(batch)}  Rows returned {len(batch[0])}  bits: {label.shape}  ")
    print(f"+-----+----------------+--------------------------+------------------------------+-----+-----+--------------------------------------------------------+")
    print(f"| idx |    batch[0]    |      batch[1]            |                                                batch[2]                                           | ") 
    print(f"|     |   COMPOUND_ID  |       CMPHASH / BIN      |                                                FEATURES                                           | ")
    print(f"+-----+----------------+--------------------------+------------------------------+-----+-----+--------------------------------------------------------+")
         ###    0 | source_11 Batch2    EC000046  K04      | JCP2022_009278 |  7406361908543180200 -  8  |   0   |   62.78000    4.13964   1.79782 | [-0.4377299 -0.4474466  1.1898487  0.2051901]
         # "  1 | source_10    | JCP2022_006020 | -9223347314827979542 |   10 |  0 | tensor([-0.6346, -0.6232, -1.6046])"

    for i in range(len(label)):
        print(f"| {i:3d} | {compound_id[i]:14s} | {cmphash[i,0]:20d}  {cmphash[i,1]:2d} | {label[i]}  |")



def define_inference_datasets(cellpainting_args, ae_runmode, ae_datetime, num_cols, ae_ckpttype, input_path, tpsa_threshold = 100):
    # ALL_INPUT_FILE   = f"3smpl_prfl_embedding_{num_cols}_HashOrder_{ae_runmode}_{ae_datetime}_{ae_ckpttype}_312000.csv"
    TRAIN_INPUT_FILE = f"3smpl_prfl_embedding_{num_cols}_HashOrder_{ae_runmode}_{ae_datetime}_{ae_ckpttype}_training.csv"
    TEST_INPUT_FILE  = f"3smpl_prfl_embedding_{num_cols}_HashOrder_{ae_runmode}_{ae_datetime}_{ae_ckpttype}_test.csv"
    VAL_INPUT_FILE   = f"3smpl_prfl_embedding_{num_cols}_HashOrder_{ae_runmode}_{ae_datetime}_{ae_ckpttype}_validation.csv"

    TRAIN_INPUT = os.path.join(input_path, TRAIN_INPUT_FILE)
    TEST_INPUT  = os.path.join(input_path, TEST_INPUT_FILE)
    VAL_INPUT   = os.path.join(input_path, VAL_INPUT_FILE)

    print(f" TRAINING INPUT   : {TRAIN_INPUT}")
    print(f" VALIDATION_INPUT : {VAL_INPUT }")
    print(f" TEST INPUT       : {TEST_INPUT }")

    cellpainting_args['training_path'] = TRAIN_INPUT
    cellpainting_args['validation_path'] = VAL_INPUT
    cellpainting_args['test_path'] = TEST_INPUT
    cellpainting_args['tpsa_threshold'] = tpsa_threshold

    #### Load CellPainting Dataset
    # data : keys to the dataset settings (and resulting keys in output dictionary)
    dataset = dict()
    data_loader = dict()

    print(f" load {dataset}")
    for datatype in ['train', 'val', 'test']:
        dataset[datatype] = CellpaintingDataset(type = datatype, **cellpainting_args)
        data_loader[datatype] = InfiniteDataLoader(dataset = dataset[datatype], batch_size=1, shuffle = False, num_workers = 0,
                                                   collate_fn = partial(dynamic_collate_fn, tpsa_threshold = dataset[datatype].tpsa_threshold) )
    return data_loader


def define_profile_datasets(cellpainting_args, num_cols, input_path, tpsa_threshold = 100):
    TRAIN_INPUT_FILE = f"3sample_profiles_{num_cols}_HashOrder_training_277200.csv"
    VAL_INPUT_FILE   = f"3sample_profiles_{num_cols}_HashOrder_validation_21600.csv"
    TEST_INPUT_FILE  = f"3sample_profiles_{num_cols}_HashOrder_test_12600.csv"

    TRAIN_INPUT = os.path.join(input_path, TRAIN_INPUT_FILE)
    TEST_INPUT  = os.path.join(input_path, TEST_INPUT_FILE)
    VAL_INPUT   = os.path.join(input_path, VAL_INPUT_FILE)

    print(f" TRAINING INPUT   : {TRAIN_INPUT}")
    print(f" VALIDATION_INPUT : {VAL_INPUT }")
    print(f" TEST INPUT       : {TEST_INPUT }")

    cellpainting_args['training_path'] = TRAIN_INPUT
    cellpainting_args['validation_path'] = VAL_INPUT
    cellpainting_args['test_path']      = TEST_INPUT
    cellpainting_args['tpsa_threshold'] = tpsa_threshold

    #### Load CellPainting Dataset
    # data : keys to the dataset settings (and resulting keys in output dictionary)
    data_loader = dict()

    for datatype in ['train', 'val', 'test']:
        print(f" load {datatype}")
        _dataset = CellpaintingDataset(type = datatype, **cellpainting_args)
        data_loader[datatype] = InfiniteDataLoader(dataset = _dataset, batch_size=1, shuffle = False, num_workers = 0,
                                                   collate_fn = partial(dynamic_collate_fn, tpsa_threshold = _dataset.tpsa_threshold) )

    return data_loader


def plot_cls_metrics(y_true, y_prob, y_pred, title = '', epochs = -1, size = 8 ):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _y_true = y_true.squeeze()
    _y_prob = y_prob.squeeze()
    _y_pred = y_pred.squeeze()
    msg_sfx = f"- ep:{epochs} " if epochs != -1 else ""

    roc_display = skm.RocCurveDisplay.from_predictions(
        _y_true,
        _y_prob,
        name=f"ROC Curve",
        color="darkorange",
        plot_chance_level=True,
        ax = axes[0])

    _ = roc_display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate")
    _ = roc_display.ax_.set_title(f" ROC curve {title.capitalize()} {msg_sfx}", size =size)
    # _ = roc_display.ax_.set_title(f"ROC curve - TPSA Classification  \n LogLoss: {metrics['logloss'] :0.3f} AUC: {metrics['roc_auc']:0.3f} ",
    _ = roc_display.ax_.legend(fontsize=8)

    pr_display = skm.PrecisionRecallDisplay.from_predictions(
        _y_true,
        _y_prob,
        name="Precision/Recall",
        pos_label= 1,
        plot_chance_level=True,
        ax = axes[1])

    _ = pr_display.ax_.set_title(f" Precision-Recall curve  {title.capitalize()} {msg_sfx}", size = size)
    _ = pr_display.ax_.legend(fontsize=8)


    cm_display = skm.ConfusionMatrixDisplay.from_predictions(
        _y_true,
        _y_pred,
        values_format="5d",
        ax = axes[2],
        colorbar = False)     # Deactivate default colorbar

    _ = cm_display.ax_.set_title(f" Confusion Matrix {title.capitalize()} {msg_sfx}", size = size)

    # Adding custom colorbar
    cax = fig.add_axes([axes[2].get_position().x1+0.01,axes[2].get_position().y0, 0.02,axes[2].get_position().height])
    plt.colorbar(cm_display.im_,  cax=cax)

    # plt.colorbar(cm_display.im_, fraction=0.046, pad=0.04 )


def plot_N_model_cls_metrics(model_outputs, names, title = '', epochs = -1, size = 9):
    fig, axes = plt.subplots(1, 2, figsize=(12,6) )
    msg_sfx = f"- epoch:{epochs} " if epochs != -1 else ""

    for model_output, name in zip(model_outputs, names):
        roc_display = skm.RocCurveDisplay.from_predictions(
            model_output.y_true.squeeze(),
            model_output.y_prob.squeeze(),
            name=name,
            # color="darkorange",
            # plot_chance_level=True,
            ax = axes[0])

        pr_display = skm.PrecisionRecallDisplay.from_predictions(
            model_output.y_true.squeeze(),
            model_output.y_prob.squeeze(),
            name=name,
            pos_label= 1,
            # plot_chance_level=True,
            ax = axes[1])

    _ = roc_display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate")
    _ = roc_display.ax_.set_title(f" ROC curve - {title} {msg_sfx}", size = size)
    _ = roc_display.ax_.legend(fontsize=8)
    # _ = pr_display.ax_.set_title(f"ROC curve - TPSA Classification  \n LogLoss: {metrics['logloss'] :0.3f} AUC: {metrics['roc_auc']:0.3f} ")

    _ = pr_display.ax_.set_title(f" Precision-Recall curve - {title} {msg_sfx}", size = size)
    _ = pr_display.ax_.legend(fontsize=8)


def run_model_on_test_data(model, data_loader, device = None, num_batches = np.inf, title = '', verbose = False):
    """
    embedding layer: layer that contains embedding (for encoding models)
    """
    out = SimpleNamespace()
    out.tpsa = np.empty((0,3))
    # out.logits = np.empty((0))
    out.compounds = np.empty((0))
    out.embeddings = {}

    out.y_true = np.empty((0,1), dtype=np.uint8)
    out.y_logits = np.empty((0,1))
    out.y_prob = np.empty((0,1))
    out.y_pred = np.empty((0,1), dtype=np.uint8)
    # out.labels = np.empty((0,1))
    if device is not None:
        model = model.to(device)
    _ = model.eval()

    for idx, (batch_features, batch_labels, batch_wellinfo, batch_compound_ids, batch_hashbin, batch_tpsa, batch_orig_labels) in enumerate(data_loader):
        #    (batch_features, batch_labels, plate_well, compound, hash)
        batch_features = batch_features.to(device)
        batch_labels   = batch_labels.to(device)
        batch_logits = model(batch_features)
        # print(f" batchtpsa : {batch_tpsa.shape}  \n {batch_tpsa[:3,:]}")
        # print(f" batchlabels: {batch_labels.shape}  batch_output: {batch_output.shape}    batch_features: {batch_features.shape}")
        out.compounds = np.concatenate((out.compounds, batch_compound_ids))
        out.y_true = np.concatenate((out.y_true, batch_labels.detach().cpu().numpy().astype(np.uint8)))
        out.y_logits = np.concatenate((out.y_logits, batch_logits.detach().cpu().numpy()))
        out.y_prob = np.concatenate((out.y_prob, torch.sigmoid(batch_logits).detach().cpu().numpy())) 

        # out.labels = np.concatenate((out.labels, batch_labels.detach().cpu().numpy()))
        out.tpsa   = np.concatenate((out.tpsa, batch_tpsa))
        if verbose:
            print(f" output :  {idx:2d} - Labels:{out.y_true.shape[0]:5d}   y_true:{out.y_true.shape}   y_pred:{out.y_pred.shape} ")
        if (idx+1) >= num_batches:
            break
    # end
    out.y_true = out.y_true.squeeze()
    out.y_logits = out.y_logits.squeeze()
    out.y_prob = out.y_prob.squeeze()
    out.y_pred = np.round(out.y_prob).astype(np.uint8)
    out.comp_labels = np.arange(out.y_true.shape[0],dtype=np.int16)//3 
    if verbose:
        print(f" {title.capitalize()} Data total batches inferred: {idx+1}")
        print(f" out.y_true     :  {out.y_true.dtype}  shape: {out.y_true.shape} sum: {out.y_true.sum()} - {out.y_true.squeeze()[:5]}")
        print(f" out.y_logits   :  {out.y_logits.dtype}  shape: {out.y_logits.shape} - {out.y_logits.squeeze()[:5]}")
        print(f" out.y_prob     :  {out.y_prob.dtype}  shape: {out.y_prob.shape} - {out.y_prob.squeeze()[:5]}")
        print(f" out.y_pred     :  {out.y_pred.dtype}  shape: {out.y_pred.shape} sum: {out.y_pred.sum()} - {out.y_pred.squeeze()[:5]}")
        print(f" out.compounds  :  {out.compounds.dtype}    shape: {out.compounds.shape}")
        print(f" out.comp_labels:  {out.comp_labels.dtype}   shape: {out.comp_labels.shape} - {out.comp_labels[:25]}", flush=True)
    return out

def train(model, optimizer, dataloader, current_epoch = 0, total_epochs = 0, device = None):
    loss = 0
    accuracy = 0
    minibatch_size = dataloader.dataset.sample_size * dataloader.dataset.compounds_per_batch
    train_minibatches = len(dataloader) // minibatch_size
    model.train()
    t_trn = tqdm.tqdm(enumerate(dataloader), initial=0, total = train_minibatches,
                      position=0, file=sys.stdout,
                      leave= False, desc=f" Trn {current_epoch}/{total_epochs}")

    for batch_count, (batch_features, y_true, _, _, _, _, _) in t_trn:
        batch_features = batch_features.to(device)
        y_true = y_true.to(device)

        # forward pass
        y_logits = model(batch_features)
        y_pred = torch.round(torch.sigmoid(y_logits))

        # loss_bce = F.binary_cross_entropy_with_logits(logits, batch_labels) ## <--- INCLUDES sigmoid
        loss_bce = loss_fn(y_logits, y_true)
        loss += loss_bce

        acc = accuracy_fn(y_true, y_pred )
        accuracy += acc

        # print(f" {batch_count} logits:  min: {logits.min()}   max: {logits.max()}  loss: {loss_bce}")
        t_trn.set_postfix({'Loss': f"{loss_bce:.4f}", 'Acc': f"{acc:.2f}", 'lbls': f"{y_true.sum()}"})

        optimizer.zero_grad(set_to_none=True)
        loss_bce.backward()
        optimizer.step()

    loss /= (batch_count + 1)
    accuracy /= (batch_count + 1)
    return loss, accuracy

    
@torch.no_grad()
def validation(model, dataloader, current_epoch = 0, total_epochs = 0, device = None):
    loss = 0
    accuracy = 0 
    minibatch_size = dataloader.dataset.sample_size * dataloader.dataset.compounds_per_batch
    val_minibatches = len(dataloader) // minibatch_size

    model.eval()
    t_val = tqdm.tqdm(enumerate(dataloader), initial=0, total = val_minibatches, 
                      position=0, file=sys.stdout, leave= False, desc=f" Val {current_epoch}/{total_epochs}") 

    for batch_count, (batch_features, y_true, _, _, _, _, _) in t_val:
        batch_features = batch_features.to(device)
        y_true = y_true.to(device)
        y_logits = model(batch_features)
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss_bce = loss_fn(y_logits, y_true)
        loss += loss_bce
        acc = accuracy_fn(y_true, y_pred)
        accuracy += acc
        t_val.set_postfix({'Loss': f"{loss_bce:.4f}", 'Acc': f"{acc:.2f}", 'lbls': f"{y_true.sum()}"})

    loss /= (batch_count+1)
    accuracy /= (batch_count+1)
    return loss, accuracy


loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def fit(model, optimizer, scheduler, data_loader, metrics, start_epoch, end_epoch, device, ckpt_file, ckpt_path):

    for epoch in range(start_epoch, end_epoch):
        trn_loss, trn_acc = train(model, optimizer, data_loader['train'], epoch, end_epoch, device)

        val_loss, val_acc = validation(model, data_loader['val'], epoch, end_epoch, device)

        metrics['loss_trn'].append(trn_loss.item())
        metrics['acc_trn'].append(trn_acc)
        metrics['loss_val'].append(val_loss.item())
        metrics['acc_val'].append(val_acc)

        scheduler.step(val_loss)

        s_dict = scheduler.state_dict()
        print(f" {datetime.now().strftime('%X')} | Ep: {epoch+1:3d}/{end_epoch:4d} | Trn loss: {trn_loss:9.6f} - Acc: {trn_acc:.4f} |"
              f" Val loss: {val_loss:9.6f} - Acc: {val_acc:.4f} |" \
              f" last_lr: {s_dict['_last_lr'][0]:.5e}  bad_ep: {s_dict['num_bad_epochs']:d}  cdwn: {s_dict['cooldown_counter']:d} ")

        if (epoch+1) % 100 == 0:
            save_checkpoint(epoch+1, model, optimizer, scheduler, metrics = metrics,
                            filename = ckpt_file.format(ep=epoch+1),
                            ckpt_path = ckpt_path, verbose = False)

    return metrics


def build_model(type, input = 0, hidden_1 = 0, hidden_2 = 0, hidden_3 = 0, device = None, verbose = True):
    assert type in ['single_layer', 'batch_norm', 'relu'], f" Invalid model type  {type}"
    # hierarchical network : nn.Linear(n_hidden_2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

    if type == "single_layer":
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_1, 1),)
    elif type == 'batch_norm':
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.BatchNorm1d(hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2, bias=True),
            nn.BatchNorm1d(hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, hidden_3, bias=True),
            nn.BatchNorm1d(hidden_3),
            nn.Tanh(),
            nn.Linear(hidden_3, 1),)
    elif type == 'relu':
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_3, 1),)

    model.to(device)
    if verbose: 
        SUMMARY_COL_NAMES = ["input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"]
        _ = summary(model, verbose = 2, input_size=(30, input), col_names = SUMMARY_COL_NAMES)
    return model

    
def build_pharmacophore_model(type, input = 0, hidden_1 = 0, hidden_2 = 0, hidden_3 = 0, output= 1, device = None, verbose = True):
    assert type in ['single_layer', 'batch_norm', 'relu'], f" Invalid model type  {type}"
    # hierarchical network : nn.Linear(n_hidden_2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

    if type == "single_layer":
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_1, 1),)
    elif type == 'batch_norm':
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.BatchNorm1d(hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2, bias=True),
            nn.BatchNorm1d(hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, hidden_3, bias=True),
            nn.BatchNorm1d(hidden_3),
            nn.Tanh(),
            nn.Linear(hidden_3, 1),)
    elif type == 'relu':
        model = nn.Sequential(
            nn.Linear(input, hidden_1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_3, output),)

    model.to(device)
    if verbose: 
        SUMMARY_COL_NAMES = ["input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"]
        _ = summary(model, verbose = 2, input_size=(30, input), col_names = SUMMARY_COL_NAMES)
    return model


    
# @torch.no_grad()
# def validation_mse(val_steps=50):
#     loss = 0
#     for i in range(val_steps):
#         ix = torch.randint(0, val_X.shape[0], (batch_size,))
#         Xv, Yv = torch.Tensor(val_X[ix]).to(device), torch.Tensor(val_y[ix]).to(device) # batch X,Y
#         logits = model(Xv)
#         loss += F.mse_loss(logits, Yv)
#     loss /= val_steps
#     return loss 

# evaluate the loss
# @torch.no_grad() # this decorator disables gradient tracking inside pytorch
# def split_loss(split):
#     """
#     compute loss for data split passed (training, validation, or test data)
#     """
#     # from torch.torcheval.metrics import R2Score    
#     # import torchmetrics 
#     from torchmetrics.regression import R2Score, PearsonCorrCoef
#     x_numpy,y_numpy = {
#     'train': (train_X, train_y),
#     'val'  : (val_X  , val_y),
#     'test' : (test_X , test_y),
#     }[split]
#     x = torch.Tensor(x_numpy).to(device)
#     y = torch.Tensor(y_numpy).to(device) 
#     logits = model(x)
# #     print(f" size of logits: {logits.shape}   size of y: {y.shape}")
#     mse_loss = F.mse_loss(logits, y)
#     r2score = R2Score().to(device)
#     pearson = PearsonCorrCoef(num_outputs=1).to(device)
#     r2_loss = r2score(logits, y) 
#     pearson_loss= pearson(logits.view(-1), y.view(-1))
#     print(f"\n {split:5s} data:   MSE loss: {mse_loss.item():10.4f}    R2 Score: {r2_loss.item():.5f}     Pearson Coeff. {pearson_loss:.4f}")

# @torch.no_grad()
# def calc_loss(x,y):
#     logits = model(x)
#     loss = F.mse_loss(logits, y)
#     print(y[:20].T)
#     print(logits[:20].T)
#     print(f"Calculated loss:  {loss.item():5e}")
