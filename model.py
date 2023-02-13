import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """

        self.num_hidden_layers = 3
        self.n_steps = n_steps
        self.n_dim = n_dim

        def getTimeEmbedding(t, dim_time=10):
            x = torch.arange(1, 1 + dim_time//2)
            x = 2*x/dim_time
            sin_emb = torch.sin(t[:,None]/10000*(x[None,:]))
            cos_emb = torch.cos(t[:,None]/10000*(x[None,:]))
            out = torch.zeros(t.shape[0],dim_time)
            even = 2*torch.arange(dim_time//2)
            out[:,even] = sin_emb
            out[:,even+1] = cos_emb
            return out

        self.time_embed = getTimeEmbedding
        self.model = torch.nn.Sequential(
            nn.Linear(self.n_dim+10, self.n_dim),
            nn.ReLU(),
            *[nn.Linear(self.n_dim, self.n_dim) if i%2==0 else nn.ReLU() for i in range(2*self.num_hidden_layers)],
            nn.Linear(self.n_dim, self.n_dim)
        )

        """
        Be sure to save at least these 2 parameters in the model instance.
        """

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        t_embed = self.time_embed(t)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        def getBeta(t):
            return lbeta+(ubeta-lbeta)*t/self.n_steps

        def getAlpha(t):
            product = 1
            for i in range(t):
                product = product*(1-getBeta(i))
            return product
        
        self.getBeta = getBeta
        self.getAlpha = getAlpha

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        """
        pass

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        t = torch.randint(0, self.n_steps, (batch.shape[0],)) # random time step
        alpha = torch.tensor([self.getAlpha(time) for time in t])[:,None] # check
        # x0 = batch[int(torch.randint(0, batch.shape[0], (1,))), :] # random batch sample
        # epsilon = torch.normal(mean=torch.FloatTensor([0.0]*batch.shape[0]), std=torch.FloatTensor([1.0]*batch.shape[1])) # sampled from N(0, I)
        epsilon = torch.randn(batch.shape)
        loss = epsilon - self.forward((alpha**0.5 * batch) + ((1-alpha)**0.5 * epsilon), t)
        # print(torch.norm(loss)**2)
        return torch.sum(loss**2)

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        X = torch.zeros((n_samples, self.n_dim, self.n_steps+1))
        X[:,:,self.n_steps] = torch.randn((n_samples, self.n_dim))
        for t in range(self.n_steps,0,-1):
            z = torch.randn((n_samples, self.n_dim))
            alpha = self.getAlpha(t)
            beta = self.getBeta(t)
            x = X[:,:,t]
            X[:,:,t-1] = (x - ((beta)/(1-alpha)**0.5)*self.forward(x,t))/(1-beta)**0.5 +z* beta**0.5
        if return_intermediate:
            return X[:,:,0], X[:,:,1:]
        else:
            return X[:,:,0]


    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.8)
