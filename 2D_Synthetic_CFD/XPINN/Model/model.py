
from Model.embedding import *
from Model.layer import * 


class PINN(nn.Module):
    def __init__(self,in_l:int=2, out_l:int=3,
                 emb_s:int=256, d:int=8, 
                 layer_mode:str='Sin',
                 _O:int=10,
                 enc_mode:bool=False,
                 L:int=10):
        super().__init__()
        self.if_pos = enc_mode
        self.init_w = False
        
        if enc_mode:
            print("Use Positional Encoding")
            in_l = in_l * ((2 *L)+1)
            self.PE = posemb(L)
            self.if_pos = True
        
        if layer_mode == 'Tanh':
            print("Tanh PINN")
            self.NN_init = TanhLayer(in_l,emb_s)
            self.NN = TanhLayer(emb_s,emb_s)
        elif layer_mode == 'ReLU':
            print("ReLU PINN")
            self.NN_init = ReLULayer(in_l,emb_s)
            self.NN = ReLULayer(emb_s,emb_s)
        elif layer_mode == 'Sin':
            print("Sin PINN")
            self.NN_init = SineLayer(in_l,emb_s,True,True,_O)
            self.NN = SineLayer(emb_s,emb_s,omega_0=_O)
            self.init_w = True
        else:
            print('None type Error')
            
        self.NN_out = nn.Linear(emb_s,out_l)
        self.depth = d
        
        if self.init_w :
            with torch.no_grad() :
                self.NN_out.weight.uniform_(-np.sqrt(6 / emb_s) / _O, np.sqrt(6 / emb_s) / _O) 
        
    def forward(self,x,y):
        inputs = torch.cat([x,y],-1)
        
        if self.if_pos:
            inputs = self.PE(inputs)
        
        out = self.NN_init(inputs)
        
        for i in range (self.depth-1):
            out = self.NN(out)
        
        out = self.NN_out(out)
        
        return out 