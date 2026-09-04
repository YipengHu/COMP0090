"""Build the six source notebooks from readable cell definitions."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def cell_id(text): return hashlib.sha1(text.encode()).hexdigest()[:12]
def md(text): return {"cell_type":"markdown","id":cell_id("m"+text),"metadata":{},"source":text.splitlines(True)}
def code(text): return {"cell_type":"code","id":cell_id("c"+text),"execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}
def write(folder, cells):
    path=ROOT/folder/"activity.ipynb"; path.parent.mkdir(parents=True,exist_ok=True)
    cells = [cells[0], md(SELF_PACED), md(NOTATION[folder]), *cells[1:], md(EXPECTED[folder])]
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
    path.write_text(json.dumps(nb,indent=1)+"\n")

common="""import numpy as np
import matplotlib.pyplot as plt
from comp0090 import seed_everything

# Fix every random-number generator so that your plots match the reference run.
# After completing the guided activity, change the seed to test robustness.
rng = seed_everything(7)

# The default path is designed for a CPU. Set this to False only after the
# notebook works and you want to run longer variants.
FAST_MODE = True"""

SELF_PACED = """## How to work through this activity

This is a guided investigation rather than a coding tutorial. For each experiment:

1. Read the mathematical claim and identify the quantity being measured.
2. Predict the qualitative result before running the code.
3. Run one cell at a time and inspect both values and plots.
4. Change only the suggested variable; rerun and explain what changed.
5. Answer the **Explain** questions in your own words.

The code contains more comments than production software intentionally. You are not expected to memorise framework syntax. Focus on the relationship between assumptions, measurements and conclusions."""

NOTATION = {
"tia_01_depth": r"""## Notation and prediction

For layer $l$, let $h_l\in\mathbb{R}^{n_l}$ be its activation, $W_l$ its weight matrix, $z_l=W_lh_l$ its pre-activation and $h_{l+1}=\phi(z_l)$. Backpropagation repeatedly applies

$$\frac{\partial L}{\partial h_l}=W_l^\top\operatorname{diag}\!\left(\phi'(z_l)\right)\frac{\partial L}{\partial h_{l+1}}.$$

Products of factors smaller or larger than one can therefore make gradients vanish or explode. Under independent, zero-mean assumptions, $\operatorname{Var}(z_l)\approx n_l\operatorname{Var}(W_l)\operatorname{Var}(h_l)$. Predict the weight variance that preserves scale for a linear activation and for ReLU.""",
"tia_02_generalisation": r"""## Notation and prediction

Given training sample $S=\{(x_i,y_i)\}_{i=1}^n$, empirical risk is

$$\hat R_S(f)=\frac1n\sum_{i=1}^n\ell(f(x_i),y_i),$$

whereas population risk is $R(f)=\mathbb E_{(x,y)\sim p_{\mathrm{data}}}[\ell(f(x),y)]$. Their difference is the generalisation gap. Parameter count measures capacity only imperfectly, and interpolation ($\hat R_S\approx0$) does not imply $R(f)\approx0$. Predict what will happen when $y_i$ is replaced by a fixed random permutation.""",
"tia_03_inductive_bias": r"""## Notation and prediction

An inductive bias favours some functions before the data have uniquely determined one. For translation operator $T_\delta$, a feature map $f$ is translation-equivariant when

$$f(T_\delta x)=T_\delta f(x),$$

and a classifier $g$ is invariant when $g(T_\delta x)=g(x)$. Convolution shares a local kernel across positions; global pooling can turn equivariant features into an invariant decision. Predict which model will extrapolate from centred bars to displaced bars.""",
"tia_04_self_supervision": r"""## Notation and prediction

Two stochastic views $x_i^{(1)}=T_1(x_i)$ and $x_i^{(2)}=T_2(x_i)$ form a positive pair. With normalised representations $z_i=f_\theta(x_i)$, the InfoNCE loss for one direction is

$$\ell_i=-\log\frac{\exp(z_i^{(1)\top}z_i^{(2)}/\tau)}{\sum_j\exp(z_i^{(1)\top}z_j^{(2)}/\tau)},$$

where $\tau$ is temperature. The augmentation distribution defines which information should become invariant. Predict what happens if a positive pair can receive transformations that alter its semantic class.""",
"tia_05_pretraining": r"""## Notation and prediction

Write a pretrained network as $f(x)=h_\psi(g_\theta(x))$, with encoder $g_\theta$ and source-task head $h_\psi$. A frozen linear probe learns only

$$\hat y=\arg\max_k\left(Wg_\theta(x)+b\right)_k,$$

so probe performance measures how linearly accessible downstream information already is in the representation. Pretraining examples and downstream test examples are separated below. Predict how random, narrow pretrained and wider pretrained encoders will behave when labels are scarce.""",
"tia_06_diffusion": r"""## Notation and prediction

The variance-preserving forward process has a closed-form marginal

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

Its score $s_t(x)=\nabla_x\log p_t(x)$ points toward increasing log-density. For the SDE $dx=-\tfrac{\beta}{2}x\,dt+\sqrt\beta\,dW_t$, reverse-time dynamics use the score to undo diffusion. Predict how the score field and sample quality change with $t$ and with the number of numerical steps.""",
}

EXPECTED = {
"tia_01_depth": """## Expected pattern and limits

Small scales collapse, large scales explode, and He scaling keeps ReLU activation variance roughly stable. The residual experiment uses a small centred branch and should preserve both activation and gradient scale better than the plain deep map. This does not establish better generalisation, nor guarantee that every residual parameterisation is stable.""",
"tia_02_generalisation": """## Expected pattern and limits

The wider model should fit arbitrary labels while remaining near chance on genuine test labels. Genuine labels should generalise, and rotation should reduce accuracy. Regularisation need not improve this already-small clean problem. The experiment demonstrates possibility—not a complete theory of neural-network generalisation.""",
"tia_03_inductive_bias": """## Expected pattern and limits

The CNN should generalise from centred to displaced bars with far fewer parameters, and its interior feature-map equivariance error should be small. Global pooling then deliberately fails when absolute position defines the label. Convolution helps because its symmetry matches the first task, not because CNNs dominate every image problem.""",
"tia_04_self_supervision": """## Expected pattern and limits

The good contrastive encoder should improve linear-probe accuracy and nearest-neighbour agreement over a random encoder. A transformation that sometimes changes bar orientation should damage the class structure. This toy instance-discrimination result is not a reproduction of full SimCLR.""",
"tia_05_pretraining": """## Expected pattern and limits

Frozen pretrained features—especially from the wider encoder—should beat random features and become useful with few downstream labels. Increasing width also increases parameters; three tiny models cannot establish a scaling law. The optional modern-model comparison should control preprocessing and probing protocol.""",
"tia_06_diffusion": """## Expected pattern and limits

Forward samples approach isotropic noise. Reverse samples recover two balanced modes, and finer discretisation places samples closer to the target mixture centres. Here the exact score replaces a trained network so the sampling mechanism is visible; real diffusion models must estimate the score or equivalent noise target.""",
}

write("tia_01_depth",[
md("""# TiA 1 — Depth, signal propagation and trainability

**Big question:** What actually goes wrong as a neural network becomes deep?

By the end, you should be able to connect products of Jacobians to vanishing/exploding signals, predict a useful initialisation scale, and explain what a residual path changes. Change one variable at a time and write a claim supported by each plot."""),
code(common),
md("""## A. Watch signals propagate

For $h_{l+1}=\phi(W_lh_l)$, mean-field reasoning predicts that the variance is repeatedly multiplied by a factor involving fan-in, weight variance and the activation derivative. Compare arbitrary scaling with Xavier ($1/n$) and He ($2/n$)."""),
code("""def propagate(depth=60, width=128, activation="relu", scale=1.0):
    h=rng.normal(size=(512,width)); variances=[]; saturated=[]
    for _ in range(depth):
        W=rng.normal(0,scale/np.sqrt(width),(width,width)); z=h@W
        h={"relu":lambda x:np.maximum(x,0),"tanh":np.tanh,"sigmoid":lambda x:1/(1+np.exp(-x))}[activation](z)
        variances.append(h.var()); saturated.append(np.mean(np.abs(h)>0.95) if activation=="tanh" else 0)
    return np.array(variances),np.array(saturated)

fig,ax=plt.subplots(1,2,figsize=(10,3))
for label,act,scale in [("small sigmoid","sigmoid",0.2),("large tanh","tanh",2.0),("ReLU: Xavier","relu",1.0),("ReLU: He","relu",np.sqrt(2))]:
    v,s=propagate(activation=act,scale=scale); ax[0].semilogy(v+1e-14,label=label)
ax[0].set(xlabel="layer",ylabel="activation variance"); ax[0].legend(fontsize=8)
for scale in [0.5,1.0,np.sqrt(2),2.0]:
    v,_=propagate(activation="relu",scale=scale); ax[1].semilogy(v+1e-14,label=f"scale={scale:.2f}")
ax[1].set(xlabel="layer",ylabel="variance"); ax[1].legend(fontsize=8); plt.show()"""),
code("""# A numerical self-check complements the log-scale plot.
# Stable here means that the last-layer variance remains within one order of
# magnitude of the first; it is not a universal guarantee for trained nets.
he_variance,_=propagate(activation="relu",scale=np.sqrt(2))
small_variance,_=propagate(activation="relu",scale=.5)
large_variance,_=propagate(activation="relu",scale=2.)
print(f"He final/initial variance: {he_variance[-1]/he_variance[0]:.3f}")
print(f"Small-scale final/initial: {small_variance[-1]/small_variance[0]:.3e}")
print(f"Large-scale final/initial: {large_variance[-1]/large_variance[0]:.3e}")
assert .05 < he_variance[-1]/he_variance[0] < 20
assert small_variance[-1]/small_variance[0] < 1e-10
assert large_variance[-1]/large_variance[0] > 1e10"""),
md("""## B. Gradients and residual paths

Autograd is used only as a measuring instrument. The experiment compares a plain update with a residual update, $h_{l+1}=h_l+αF_l(h_l)$. It does **not** claim residual networks generalise better."""),
code("""import torch
def gradient_profile(depth=80,width=64,residual=False,alpha=0.1):
    # Keep references to intermediate activations so autograd exposes dL/dh_l.
    x=torch.randn(256,width,requires_grad=True); hs=[]; h=x
    for _ in range(depth):
        # tanh makes a centred residual branch; alpha controls its perturbation.
        W=torch.randn(width,width)/np.sqrt(width); z=torch.tanh(h@W)
        h=h+alpha*z if residual else z
        h.retain_grad(); hs.append(h)
    h.square().mean().backward()
    return np.array([q.detach().std().item() for q in hs]),np.array([q.grad.norm().item() for q in hs])
fig,ax=plt.subplots(1,2,figsize=(10,3))
profiles={}
for residual in [False,True]:
    a,g=gradient_profile(residual=residual); label="residual" if residual else "plain"
    profiles[label]=(a,g)
    ax[0].semilogy(a,label=label); ax[1].semilogy(g,label=label)
    print(f"{label:8s}: first/last activation std={a[0]:.3f}/{a[-1]:.3f}; minimum gradient norm={g.min():.3e}")
ax[0].set(title="activation scale",xlabel="depth"); ax[1].set(title="gradient norm",xlabel="depth")
for a in ax:a.legend();a.grid(alpha=.2)
plt.show()
assert profiles["residual"][0][-1] > 10*profiles["plain"][0][-1]
assert profiles["residual"][1].min() > 10*profiles["plain"][1].min()"""),
md("""### Explain

1. Which initialisation preserves variance for ReLU, and under what assumptions?
2. Why can stable forward activations coexist with unstable gradients?
3. What identity term appears in a residual block's Jacobian?

**Reading:** [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) and [He et al. (2016)](https://arxiv.org/abs/1512.03385).""")])

write("tia_02_generalisation",[
md("""# TiA 2 — Generalisation, memorisation and regularisation

**Big question:** How can the same overparameterised model generalise from meaningful labels and memorise random ones?

We separate fitting ability from out-of-sample performance, then perturb capacity, regularisation, and the test distribution."""),code(common),
code("""from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
X,y=load_digits(return_X_y=True); X=X/16
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.35,stratify=y,random_state=7)
# A smaller training sample makes exact memorisation observable on a laptop.
Xtr,ytr=Xtr[:600],ytr[:600]

def fit(width,labels,alpha=0.0):
    # L-BFGS is used to study attainable fits, not stochastic training dynamics.
    m=MLPClassifier((width,),alpha=alpha,solver="lbfgs",max_iter=300 if FAST_MODE else 800,random_state=7).fit(Xtr,labels)
    return m,accuracy_score(labels,m.predict(Xtr)),accuracy_score(yte,m.predict(Xte))

rows=[]
random_y=rng.permutation(ytr)
for width in [8,32,64]:
    for name,labels in [("true",ytr),("random",random_y)]:
        m,tr,te=fit(width,labels); rows.append((width,name,sum(w.size for w in m.coefs_),tr,te))
print("width labels parameters train_acc test_acc")
for r in rows: print(f"{r[0]:5} {r[1]:6} {r[2]:10} {r[3]:.3f} {r[4]:.3f}")
widest_random=[r for r in rows if r[0]==64 and r[1]=="random"][0]
widest_true=[r for r in rows if r[0]==64 and r[1]=="true"][0]
assert widest_random[3] > .95 and widest_random[4] < .20
assert widest_true[4] > .85"""),
md("""## Regularisation and shift

Weight decay changes the objective; early stopping changes the optimisation path; augmentation changes the empirical distribution. They need not have equivalent effects."""),
code("""from scipy.ndimage import rotate
model,_,_=fit(128,ytr,alpha=1e-2)
shifted=np.stack([rotate(im.reshape(8,8),18,reshape=False,mode="nearest").ravel() for im in Xte])
print({"in_distribution":accuracy_score(yte,model.predict(Xte)),"rotated":accuracy_score(yte,model.predict(shifted))})
for alpha in [0,1e-3,1e-1,1]:
    _,tr,te=fit(128,ytr,alpha); print(f"weight_decay={alpha:g}: train={tr:.3f}, test={te:.3f}")
assert accuracy_score(yte,model.predict(shifted)) < accuracy_score(yte,model.predict(Xte))-.05"""),
md("""### Explain

1. Does interpolation imply generalisation? Use the random-label control.
2. Why is the rotated set not merely a noisier estimate of the same test accuracy?
3. Identify one conclusion this small experiment cannot justify.

**Reading:** [Zhang et al., Understanding Deep Learning Requires Rethinking Generalization](https://openreview.net/forum?id=Sy8gdB9xx).""")])

write("tia_03_inductive_bias",[
md("""# TiA 3 — Inductive bias and equivariance

**Big question:** If two models are expressive enough, why can one learn an image task with far fewer examples?

We isolate locality and weight sharing using synthetic bars; no external image dataset is needed."""),code(common+"\nfrom comp0090 import make_shape_images\nimport torch\nfrom torch import nn"),
code("""Xtr,ytr=make_shape_images(120,centred=True); Xte,yte=make_shape_images(500,centred=False,seed=8)
class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(144,24),nn.ReLU(),nn.Linear(24,2))
    def forward(self,x): return self.net(x)
class CNN(nn.Module):
    def __init__(self): super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU()); self.head=nn.Linear(4,2)
    def forward(self,x): return self.head(self.features(x).mean((2,3)))
def train(model,epochs=60):
    x=torch.tensor(Xtr); y=torch.tensor(ytr); opt=torch.optim.Adam(model.parameters(),lr=.03)
    for _ in range(epochs): opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step()
    with torch.no_grad(): return (model(torch.tensor(Xte)).argmax(1).numpy()==yte).mean()
results={}
for model in [MLP(),CNN()]:
    # Both see only centred training bars; testing moves bars to unseen positions.
    results[type(model).__name__]=train(model)
    print(type(model).__name__,sum(p.numel() for p in model.parameters()),f"shifted-position accuracy={results[type(model).__name__]:.3f}")
assert results["CNN"] > .90
assert results["CNN"] > results["MLP"]+.30"""),
md("""## Measure equivariance directly

For translation $T$ and representation $f$, compute $\|f(Tx)-Tf(x)\|/\|f(x)\|$. Cropping at boundaries is excluded from the comparison."""),
code("""cnn=CNN(); x=torch.tensor(Xte[:32]); shift=lambda z:torch.roll(z,2,dims=-1)
with torch.no_grad():
    fx=cnn.features(x); lhs=cnn.features(shift(x)); rhs=shift(fx)
    interior=(lhs[:,:,:,2:-2]-rhs[:,:,:,2:-2]).norm()/rhs[:,:,:,2:-2].norm()
print(f"CNN representation equivariance error: {interior:.2e}")
assert interior < .10

# Break the assumption: label is whether the bar lies left/right of centre.
Xp,_=make_shape_images(500,seed=12); pos=Xp[:,0].sum(1).argmax(1); yp=(pos>=6).astype("int64")
print("Global average pooling deliberately discards the absolute position needed by this task.")"""),
md("""### Explain

1. Distinguish invariance from equivariance.
2. Why does global pooling help the first task and hurt the absolute-position task?
3. State the conditions under which convolution improves sample efficiency.

**Reading:** [Bronstein et al., Geometric Deep Learning](https://arxiv.org/abs/2104.13478), Sections 2–3.""")])

write("tia_04_self_supervision",[
md("""# TiA 4 — Self-supervised representation learning

**Big question:** Can a useful representation emerge without class labels?

This intentionally tiny contrastive experiment tests the effect of augmentation assumptions. Labels are used only after representation learning, in a linear probe."""),code(common+"\nfrom comp0090 import make_shape_images\nimport torch\nfrom torch import nn\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.decomposition import PCA"),
code("""X,y=make_shape_images(800,seed=20); x=torch.tensor(X)
class Encoder(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(144,48),nn.ReLU(),nn.Linear(48,16))
    def forward(self,z): return nn.functional.normalize(self.net(z),dim=1)
def augment(z,bad=False):
    # Translation and noise preserve bar orientation, hence the class label.
    out=torch.roll(z,int(rng.integers(-2,3)),dims=-1)+.08*torch.randn_like(z)
    # In the bad condition, each view is independently transposed with 50%
    # probability. A positive pair can therefore disagree in orientation.
    return out.transpose(-1,-2) if bad and rng.random()<.5 else out
def contrastive_train(bad=False):
    enc=Encoder(); opt=torch.optim.Adam(enc.parameters(),lr=2e-3); epochs=12 if FAST_MODE else 30
    for _ in range(epochs):
        # The two calls generate independent views of the same source examples.
        ids=torch.randperm(len(x))[:256]; a,b=enc(augment(x[ids],bad)),enc(augment(x[ids],bad))
        # Every row's matching column is its positive; other columns are negatives.
        logits=a@b.T/.15; target=torch.arange(len(ids)); loss=(nn.functional.cross_entropy(logits,target)+nn.functional.cross_entropy(logits.T,target))/2
        opt.zero_grad();loss.backward();opt.step()
    return enc
def probe(features,n_labels=80):
    idx=np.r_[np.where(y==0)[0][:n_labels//2],np.where(y==1)[0][:n_labels//2]]; test=np.setdiff1d(np.arange(len(y)),idx)
    clf=LogisticRegression().fit(features[idx],y[idx]); return clf.score(features[test],y[test])
raw=X.reshape(len(X),-1); random_features=Encoder()(x).detach().numpy(); good=contrastive_train()(x).detach().numpy(); bad=contrastive_train(True)(x).detach().numpy()
probe_scores={name:probe(z) for name,z in [("pixels",raw),("random",random_features),("contrastive",good),("bad augmentation",bad)]}
for name,value in probe_scores.items(): print(f"{name:16s} linear-probe accuracy={value:.3f}")
assert probe_scores["contrastive"] > probe_scores["random"]+.20
assert probe_scores["bad augmentation"] < probe_scores["contrastive"]-.10"""),
code("""fig,ax=plt.subplots(1,2,figsize=(9,3))
for a,(name,z) in zip(ax,[("random encoder",random_features),("contrastive encoder",good)]):
    p=PCA(2).fit_transform(z); a.scatter(*p.T,c=y,s=8,cmap="coolwarm"); a.set_title(name)
plt.show()
similarity=good@good.T; np.fill_diagonal(similarity,-np.inf)
print("Nearest-neighbour label agreement:",np.mean(y[similarity.argmax(1)]==y))"""),
md("""### Explain

1. What invariance does the augmentation define?
2. Why is a linear probe evidence about representation geometry, not end-to-end performance?
3. Explain the bad-augmentation result without saying the optimiser failed.

**Reading:** [Chen et al., SimCLR](https://proceedings.mlr.press/v119/chen20j.html).""")])

write("tia_05_pretraining",[
md("""# TiA 5 — Pretraining, transfer and model scale

**Big question:** What has a pretrained model already learned, and how reusable is it?

The default experiment performs genuine pretraining and frozen-feature transfer on the packaged scikit-learn digits data. It is deliberately small and offline. The extension links to standard pretrained vision weights; it is not required to run this notebook."""),code(common+"\nimport torch\nfrom torch import nn\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler"),
code("""X,y=load_digits(return_X_y=True); X=(X/16).astype("float32"); xt=torch.tensor(X)
pretrain_idx,transfer_idx=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=7)
class Encoder(nn.Module):
    def __init__(self,width): super().__init__(); self.features=nn.Sequential(nn.Linear(64,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU()); self.head=nn.Linear(width,10)
    def forward(self,z,features=False):
        h=self.features(z); return h if features else self.head(h)
def pretrain(width,epochs=45):
    model=Encoder(width); opt=torch.optim.Adam(model.parameters(),lr=.01); target=torch.tensor(y[pretrain_idx])
    for _ in range(epochs): opt.zero_grad(); loss=nn.functional.cross_entropy(model(xt[pretrain_idx]),target); loss.backward();opt.step()
    return model
models={w:pretrain(w,25 if FAST_MODE else 60) for w in [8,32,96]}
random_model=Encoder(32)
features={"pixels":X,"random":random_model(xt,True).detach().numpy(),**{f"pretrained-{w}":m(xt,True).detach().numpy() for w,m in models.items()}}"""),
md("""## Frozen linear probes and label efficiency

The encoder is frozen. Only multinomial logistic regression sees downstream labels, isolating the information already present in each representation."""),
code("""# Only examples held out from pretraining enter this downstream split.
train,test=train_test_split(transfer_idx,test_size=.5,stratify=y[transfer_idx],random_state=7)
scores_by_count={}
for count in [5,20,60]:
    chosen=np.concatenate([train[y[train]==c][:count] for c in range(10)])
    scores={name:make_pipeline(StandardScaler(),LogisticRegression(max_iter=600)).fit(z[chosen],y[chosen]).score(z[test],y[test]) for name,z in features.items()}
    scores_by_count[count]=scores
    print(f"labels/class={count}",scores)
print("parameter counts",{w:sum(p.numel() for p in m.parameters()) for w,m in models.items()})
assert scores_by_count[5]["pretrained-96"] > scores_by_count[5]["random"]+.15
assert scores_by_count[20]["pretrained-96"] > scores_by_count[5]["pretrained-96"]"""),
md("""### Modern-model extension (optional)

Repeat the same `extract → freeze → probe` protocol with [torchvision's pretrained ResNet-18 and larger models](https://pytorch.org/vision/stable/models.html), or a [Vision Transformer](https://pytorch.org/vision/stable/models/vision_transformer.html). Those weights require an explicit download and substantially more memory, so they are not part of the default path. Record parameters, extraction time, memory, and probe accuracy—larger is a hypothesis, not a conclusion.

### Explain

1. Why can frozen features reduce labelled-data requirements?
2. What confound prevents this toy experiment from establishing a universal scaling law?
3. Distinguish pretraining, linear probing, and fine-tuning.""")])

write("tia_06_diffusion",[
md("""# TiA 6 — Diffusion: from noise to data

**Big question:** What must a model know to turn noise into data?

A two-component Gaussian mixture makes the forward marginals and their exact score visible. This lets us inspect reverse diffusion without downloading weights or hiding the mechanism in a framework."""),code(common),
code("""centres=np.array([[-2.,0.],[2.,0.]])
def alpha(t): return np.exp(-3*t)
def sample_forward(x0,t):
    a=alpha(t); return np.sqrt(a)*x0+np.sqrt(1-a)*rng.normal(size=x0.shape)
x0=centres[rng.integers(0,2,1000)]+.25*rng.normal(size=(1000,2))
fig,ax=plt.subplots(1,5,figsize=(12,2.5))
for a,t in zip(ax,np.linspace(0,1,5)):
    z=sample_forward(x0,t); a.scatter(*z.T,s=3); a.set_title(f"t={t:.2f}"); a.set_xlim(-4,4);a.set_ylim(-3,3)
plt.show()"""),
md("""## Exact score and reverse-time sampling

For this known mixture, we can calculate $\nabla_x\log p_t(x)$ exactly. A learned denoiser estimates equivalent information. Euler–Maruyama then combines the score-driven drift with stochastic noise; the network and the sampler are different objects."""),
code("""sigma0=.25
def score(x,t):
    a=alpha(t); means=np.sqrt(a)*centres; var=a*sigma0**2+1-a
    logp=-((x[:,None,:]-means[None,:,:])**2).sum(2)/(2*var)
    w=np.exp(logp-logp.max(1,keepdims=True)); w/=w.sum(1,keepdims=True)
    return (w[:,:,None]*(means[None,:,:]-x[:,None,:])/var).sum(1)
def reverse_sample(steps,n=600):
    x=rng.normal(size=(n,2)); dt=1/steps; snapshots=[x.copy()]
    # Reverse SDE for forward dX=-1.5X dt+sqrt(3)dW, integrated from t=1 to 0.
    for k in range(steps,0,-1):
        t=k/steps; x += (1.5*x+3*score(x,t))*dt+np.sqrt(3*dt)*rng.normal(size=x.shape)
        if k in {steps,3*steps//4,steps//2,steps//4,1}: snapshots.append(x.copy())
    return x,snapshots
fig,ax=plt.subplots(1,3,figsize=(9,3))
quality={}
for a,steps in zip(ax,[10,30,100]):
    z,_=reverse_sample(steps)
    # Distance to the nearest component centre is a simple transparent proxy.
    nearest_distance=np.sqrt(((z[:,None,:]-centres[None,:,:])**2).sum(2)).min(1).mean()
    quality[steps]=nearest_distance
    a.scatter(*z.T,s=3); a.set_title(f"{steps} steps\\nmean distance={nearest_distance:.2f}");a.set_xlim(-4,4);a.set_ylim(-3,3)
plt.show()
print("Mean distance to nearest target mode (lower is better):",quality)
assert quality[100] < quality[10]-.15"""),
md("""### Real-image bridge (optional)

Use the same questions—intermediate states, fixed initial noise, step count, scheduler—with a [small pretrained DDPM from Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/ddpm). It requires model-weight downloads and is therefore intentionally outside the default notebook.

### Explain

1. What does the score point toward at early versus late noise times?
2. Why can changing the sampler change outputs while the learned network stays fixed?
3. Describe the compute–quality trade-off in the step-count plot.

**Reading:** [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). For the lecture connection to flows, see [Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747).""")])

print("Built six notebooks.")
