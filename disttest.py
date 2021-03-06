#export
from fastai.test_utils import *
from fastai.callback.training import *
from fastai.callback.fp16 import *
from fastai.basics import *
from fastai.distributed import *
from torch.utils.data import TensorDataset
#export
#Tests that afer n_batches, parameters should all be the same in fp16 and fp32
class GetParameters(Callback):
    "Record non-overflowing parameters, and batches that overflowed in fp16"
    def __init__(self):
        self.overflows=[]
        self.ps=L()
    def after_backward(self):
        if grad_overflow([self.model.parameters()]): 
            self.overflows+=[int((self.iter))]
        self.ps+=to_detach(L([p.clone()[0] for p in self.model.parameters()]))
#Used to skip batches in fp32, which overflow in fp16
class Skip(Callback):
    run_after=GetParameters
    "Skips and zero_grads specified batches"
    def __init__(self,skips):
        self.skips=skips
    def after_backward(self):
        if self.iter in self.skips: 
            self.learn.opt.zero_grad()
            raise CancelBatchException()
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

with no_random(seed): 
    #to_native_fp16 will not convert default synth_learner model
    model=nn.Sequential(nn.Linear(1,1))
    db=synth_dbunch(bs=bs,n_train=n_batches,n_valid=n_batches,cuda=True)
    learn = synth_learner(model=model,data=db,cbs=[GetParameters()])
    learn.to_native_fp16()
    with learn.distrib_ctx(): learn.fit(10, lr=0.01)
pa=learn.get_parameters.ps
overflows=learn.get_parameters.overflows

with no_random(seed): 
    model=nn.Sequential(nn.Linear(1,1))
    db=synth_dbunch(bs=bs,n_train=n_batches,n_valid=n_batches,cuda=True)
    learn = synth_learner(model=model,data=db,cbs=[Skip(overflows),GetParameters()])
    with learn.distrib_ctx(): learn.fit(10, lr=0.01)
pb=learn.get_parameters.ps

#test all batches had same parameters in fp16 and fp32
print(pa)
print(pb)
test_close(pa,pb,eps=1e-3)
