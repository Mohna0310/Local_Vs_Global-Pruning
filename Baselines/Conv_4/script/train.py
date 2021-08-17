import argparse
import torch
import time
import json
import numpy as np
import math
import random,os
from pathlib import Path
import torch.nn.utils.prune as prune

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic =True

# import EarlyStopping
from pytorchtools import EarlyStopping

def batch_generator(X, y, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len=np.sum(X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_X_len.argsort()[::-1]
        batch_X_len=batch_X_len[batch_idx]
        batch_X_mask=(X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_X=X[offset:offset+batch_size][batch_idx] 
        batch_y=y[offset:offset+batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda() )
        batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda() )
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda() )
        if len(batch_y.size() )==2 and not crf:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask)
            
class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.conv1=torch.nn.Conv1d(gen_emb.shape[1], 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(dropout)

        self.conv2=torch.nn.Conv1d(128, 256, 5, padding=2)
        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)

        self.linear_ae=torch.nn.Linear(256, num_classes)
        self.crf_flag=crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf=ConditionalRandomField(num_classes)            
          
    def forward(self, x, x_len, x_mask, x_tag=None, testing=False):
        x_emb= self.gen_embedding(x)
        x_emb=self.dropout(x_emb).transpose(1, 2)
        x_conv=torch.nn.functional.relu(self.conv1(x_emb))

        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv2(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        
        x_conv=x_conv.transpose(1, 2)
        x_logit=self.linear_ae(x_conv)
        if testing:
            if self.crf_flag:
                score=self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit=x_logit.transpose(2, 0)
                score=torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score=-self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit=torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score

def valid_loss(model, valid_X, valid_y, crf=False):
    model.eval()
    losses=[]
    for batch in batch_generator(valid_X, valid_y, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask=batch
        loss=model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y)
        # losses.append(loss.data[0])
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)




def train(train_X, train_y, valid_X, valid_y, model, model_fn, optimizer, parameters, parameters_to_prune, prun_amount, epochs=200,lr_scheduler=None,batch_size=128,crf=False, pruning = 'local'):
    
    best_loss=float("inf") 
    valid_history=[]
    train_history=[]
    
    # to track the training loss as the model trains
    train_losses = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience= 100, verbose=True)
    
    for epoch in range(epochs):
        for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask=batch
            loss=model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()

    ############################################################
        if pruning == 'local':
            # Local Pruning
            for j,i in enumerate(parameters_to_prune):
                modu,parm = i
                prune.l1_unstructured(modu, name=parm, amount= prun_amount[j])
                prune.remove(modu, parm)
            
        elif pruning == 'global':    
            # Global Pruning
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount= prun_amount,)

            for j,i in enumerate(parameters_to_prune):
                modu,parm = i
                prune.remove(modu, parm)

    ############################################################

        loss=valid_loss(model, train_X, train_y, crf=crf)
        train_history.append(loss)
        train_losses.append(loss)
        
        loss=valid_loss(model, valid_X, valid_y, crf=crf)
        valid_history.append(loss)
        valid_losses.append(loss)
        
        tr_loss = np.average(train_losses)
        val_loss = np.average(valid_losses)
        



        if loss<best_loss:
            best_loss=loss
            if pruning == 'local':
                torch.save(model, model_fn+"_local.pt")  
            else:
                torch.save(model, model_fn+"_global.pt")

        shuffle_idx=np.random.permutation(len(train_X) )
        train_X=train_X[shuffle_idx]
        train_y=train_y[shuffle_idx]


    ############################################################
        
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {tr_loss:.5f} ' +
                     f'valid_loss: {val_loss:.5f}')
        
        print(print_msg)
        
        # Clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping !")
            break
        
    ##############################################################
        # Learning rate scheduler
        if lr_scheduler is not None:
            # Change the learning rate
            lr_scheduler.step(loss)


    if pruning == 'local':
        model=torch.load(model_fn+"_local.pt")  
    else:
        model=torch.load(model_fn+"_global.pt") 

    
    return train_history, valid_history


########################################################################################################

from torch.optim.lr_scheduler import ReduceLROnPlateau

class lr_scheduler_list:
    """ReduceLROnPlateau """
    def __init__(self, optimizer):
        self.lr_scheduler_list = [ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, 
                verbose=True, threshold=0.01, threshold_mode='abs')]
    
    def step(self, valid_loss):
        for scheduler in self.lr_scheduler_list:
            scheduler.step(valid_loss)
            

########################################################################################################

def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128, pruning = None, pr = None):
    #gen_emb=np.load(data_dir+"gen.vec.npy")
    gen_emb=np.load(data_dir+"glove.840B.300d.txt.npy")
    domain_emb=np.load(data_dir+domain+"_emb.vec.npy")
    ae_data=np.load(data_dir+domain+".npz")
    
    valid_X=ae_data['train_X'][-valid_split:]
    valid_y=ae_data['train_y'][-valid_split:]
    train_X=ae_data['train_X'][:-valid_split]
    train_y=ae_data['train_y'][:-valid_split]

    for r in range(runs):
        print(r)
        model=Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters, lr=lr)

        ##########################################################

        # Pruning Info

        parameters_to_prune = []
        for name,mod in list(model.named_modules())[1:]:
            
            keys = dict(mod.named_parameters()).keys()
            
            if dict().keys() == keys:
                continue
            else:
                if 'bias' in keys:
                    parameters_to_prune.append((mod,'bias'))
                
                if 'weight' in keys: 
                    parameters_to_prune.append((mod,'weight'))
            
    
        if pruning == 'local':
            prun_amount = pr
            #prun_amount = [0]*len(parameters_to_prune)

            
            
        elif pruning == 'global':
            prun_amount = pr

        lr_scheduler = lr_scheduler_list(optimizer)

        ##########################################################


        train_history, valid_history = train(train_X, train_y, valid_X, valid_y, model, model_dir+domain+str(r), optimizer, parameters, parameters_to_prune, prun_amount, \
            epochs, lr_scheduler, crf=False, pruning = pruning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.getcwd()+"/model/")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default=os.getcwd()+"/data/prep_data/")
    parser.add_argument('--valid', type=int, default=150) #number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001) # lr = 0.0001
    parser.add_argument('--dropout', type=float, default=0.45) 
    parser.add_argument('--pruning', type=str, default= 'local') 

    args = parser.parse_args()



    if args.pruning == 'local':

        pruning_frac =      [[0.13, 0, 0.20, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0],
                            [0.2, 0, 0.30, 0, 0.3, 0, 0.3, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0],
                            [0.3, 0, 0.35, 0, 0.35, 0, 0.35, 0, 0.35, 0, 0.35, 0, 0.35, 0, 0],
                            [0.4, 0, 0.45, 0, 0.45, 0, 0.45, 0, 0.45, 0, 0.45, 0, 0.45, 0, 0],
                            [0.5, 0, 0.55, 0, 0.55, 0, 0.55, 0, 0.55, 0, 0.5, 0, 0.5, 0, 0],
                            [0.6, 0.5, 0.65, 0.5, 0.65, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.2],
                            [0.7, 0.5, 0.75, 0.5, 0.75, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.2],
                            [0.8, 0.5, 0.85, 0.5, 0.85, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2],
                            [0.9, 0.75, 0.9, 0.75, 0.9, 0.75, 0.9, 0.75, 0.9, 0.75, 0.9, 0.75, 0.9, 0.75, 0.3],
                            [0.95, 0.75, 0.95, 0.750, 0.95, 0.75, 0.95, 0.750, 0.95, 0.75, 0.95, 0.75, 0.93, 0.75, 0.3]]

    else:

        pruning_frac =  [0.13, 0.22, 0.32, 0.41,0.51,0.62,0.71,0.81,0.91,0.95]

    for pr_per, pr in zip(list(range(10,91,10))+[95],pruning_frac):

        if args.pruning == 'local':
            print(f"Local Pruning : {pr_per} %")
            abs_path = Path(args.model_dir+"local_"+ args.domain+"/"+str(pr_per)+'%/')

            if not os.path.isdir(abs_path):
                abs_path.mkdir(mode = 0o007, parents= True, exist_ok= True)
                run(args.domain, args.data_dir, str(abs_path) +"/", args.valid, args.runs, args.epochs, args.lr, args.dropout, args.batch_size, args.pruning, pr)

        elif args.pruning == 'global':
            print(f"Global Pruning : {pr_per} %")
            abs_path = Path(args.model_dir+"global_"+ args.domain+"/"+str(pr_per)+'%/')

            if not os.path.isdir(abs_path):
                abs_path.mkdir(mode = 0o007, parents= True, exist_ok= True)
                run(args.domain, args.data_dir, str(abs_path) +"/", args.valid, args.runs, args.epochs, args.lr, args.dropout, args.batch_size, args.pruning, pr)

        else:
            print("Number of pruning ratio does not match with model's layer.")
            break