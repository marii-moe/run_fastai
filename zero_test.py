#export
from fastai.test_utils import *
from fastai.callback.training import *
from fastai.callback.fp16 import *
from fastai.basics import *
from fastai.distributed import *
from torch.utils.data import TensorDataset
from deepspeed.runtime.zero.stage1 import FP16_DeepSpeedZeroOptimizer_Stage1
#export
n_batches=20
bs=64
seed=1337#random.randint(0,2**32-1)

def synth_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2, cuda=False):
    def get_data(n):
        x = torch.randn(bs*n, 1)
        return TensorDataset(x, a*x + b + 0.1*torch.randn(bs*n, 1))
    train_ds = get_data(n_train)
    valid_ds = get_data(n_valid)
    device = rank_distrib() #default_device() if cuda else None
    train_dl = TfmdDL(train_ds, bs=bs, shuffle=True, num_workers=1)
    valid_dl = TfmdDL(valid_ds, bs=bs, num_workers=1)
    return DataLoaders(train_dl, valid_dl, device=device)
    
    
def synth_learner(n_trn=10, n_val=2, cuda=False, lr=1e-3,opt_func=partial(SGD, mom=0.9), data=None, model=None, **kwargs):
    if data is None: data=synth_dbunch(n_train=n_trn,n_valid=n_val, cuda=cuda)
    if model is None: model=RegModel()
    return Learner(data, model, lr=lr, loss_func=MSELossFlat(),
                   opt_func=opt_func, **kwargs)
        
@patch
def set_hypers(self:FP16_DeepSpeedZeroOptimizer_Stage1,**kwargs):
    self.optimizer.set_hypers(**kwargs)
@patch(as_prop=True)
def hypers(self:FP16_DeepSpeedZeroOptimizer_Stage1,**kwargs):
    return self.optimizer.hypers

def opt_f(params,*args,**kwargs):
    #if rank_distrib()==0: import pdb;pdb.set_trace()
    return FP16_DeepSpeedZeroOptimizer_Stage1(Adam(params,*args,**kwargs))

with no_random(seed): 
    model=nn.Sequential(nn.Linear(1,1))
    db=synth_dbunch(bs=bs,n_train=n_batches,n_valid=n_batches,cuda=True)
    learn = synth_learner(model=model,data=db, opt_func=opt_f)
    learn.model.to(learn.dls.device)
    with learn.distrib_ctx(): learn.fit(10, lr=0.01)