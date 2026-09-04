# Reading list

## Foundations and optimisation

- Goodfellow, Bengio and Courville, [*Deep Learning*](https://www.deeplearningbook.org/), especially Chapters 6–8. A broad reference for networks, backpropagation, regularisation and optimisation.
- Glorot and Bengio (2010), [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html). Core reading for TiA 1.
- He et al. (2015), [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852). Variance-preserving initialisation for rectified networks.
- He et al. (2016), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Residual paths and deep optimisation; core reading for TiA 1.
- Kingma and Ba (2015), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). A widely used adaptive optimiser.

## Generalisation and regularisation

- Zhang et al. (2017), [Understanding Deep Learning Requires Rethinking Generalization](https://openreview.net/forum?id=Sy8gdB9xx). Core reading for TiA 2.
- Srivastava et al. (2014), [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html).
- Belkin et al. (2019), [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/doi/10.1073/pnas.1903070116). Double descent and interpolation.
- Nakkiran et al. (2021), [Deep Double Descent](https://openreview.net/forum?id=B1g5sA4twr). Model, sample and epoch-wise double descent.

## Architectures and inductive bias

- LeCun et al. (1998), [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). A foundational convolutional-network paper.
- Bronstein et al. (2021), [Geometric Deep Learning](https://arxiv.org/abs/2104.13478). Symmetry, invariance and equivariance; core reading for TiA 3.
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The Transformer architecture.
- Dosovitskiy et al. (2021), [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929). Vision Transformers and architectural bias.

## Representations and self-supervision

- Chen et al. (2020), [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html). SimCLR; core reading for TiA 4.
- Oord, Li and Vinyals (2018), [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).
- He et al. (2020), [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722).
- Grill et al. (2020), [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733). A non-contrastive self-supervised method.

## Pretraining, foundation models and adaptation

- Yosinski et al. (2014), [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792). A direct foundation for TiA 5.
- Kaplan et al. (2020), [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Shows why parameter count alone is not a sufficient notion of scale.
- Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). CLIP as a modern transfer-learning reference point.
- Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9). Parameter-efficient adaptation.

## Generative learning

- Kingma and Welling (2014), [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Goodfellow et al. (2014), [Generative Adversarial Nets](https://papers.nips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html).
- Ho, Jain and Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Core reading for TiA 6.
- Song et al. (2021), [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS). Connects scores, diffusion and reverse-time SDEs.
- Lipman et al. (2023), [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). Core bridge from TiA 6 to flows.

## Selected modern research examples

These are illustrative research directions, not a claim about a permanent leaderboard; “state of the art” changes quickly.

- Peebles and Xie (2023), [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). Diffusion Transformers (DiT).
- Esser et al. (2024), [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). A large-scale flow-based image model.
- Oquab et al. (2024), [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). Large-scale self-supervised visual representations.
- Dubey et al. (2024), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). An example of modern open foundation-model development and evaluation.
