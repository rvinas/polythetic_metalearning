import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


def conv_block(in_channels, out_channels):
    # bn = CustomBatchNorm()
    bn = nn.BatchNorm2d(out_channels, momentum=0.01, track_running_stats=False)
    # nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
        )
        self.embeddings = nn.Linear(hid_dim * 3 * 3, z_dim)
        self.mean = nn.Linear(z_dim, z_dim)
        self.logvar = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        h = x.view(x.size(0), -1)
        h = self.embeddings(h)
        h = nn.ReLU()(h)
        mean = self.mean(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        return Normal(mean, std)


def count_acc(probs, label):
    if len(probs.shape) == 1:
        pred = probs > 0.5
    else:
        pred = probs.argmax(-1)
    return (pred == label).type(torch.FloatTensor).mean().item()


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def train(task_generator, forward_fn, config, model=None, xor_task=None, loss_fn=None):
    # Prepare data
    g = task_generator(config=config)
    x_dim = g.input_shape[-1]

    # Set up model
    if model is None:
        model = Convnet(x_dim=x_dim, z_dim=config.out_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tl = Averager()
    ta = Averager()
    tk = Averager()

    # Train
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    for epoch in range(1, config.max_epoch + 1):
        optimizer.zero_grad()
        model.train()

        # Get and reshape data
        data_shot, label_shot, data_query, label_query = g.get_shot_query(config, device, xor_task=xor_task,
                                                                          prob_xor=config.prob_xor)

        # Compute predictions
        data = torch.cat((data_shot, data_query), 0)
        embeddings_dist = model(data)
        embeddings = embeddings_dist.rsample()
        data_shot = embeddings[:data_shot.shape[0]]
        data_query = embeddings[data_shot.shape[0]:]
        probs = forward_fn(data_shot, data_query, label_shot, config=config)

        # Compute loss
        # print(embeddings.shape)
        prior = Normal(0, 1)  # Normal(torch.zeros_like(embeddings), torch.ones_like(embeddings))
        kl = config.beta * kl_divergence(embeddings_dist, prior).mean(dim=0).sum()
        loss = loss_fn(probs, label_query) + kl  # F.cross_entropy(logits, label_query)
        acc = count_acc(probs, label_query)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        tl.add(loss.item())
        ta.add(acc)
        tk.add(kl)

        if config.verbose and epoch % 100 == 0:
            print('epoch {}, loss={:.4f}, kl={:.4f}, acc={:.4f}'.format(epoch, tl.item(), tk.item(), ta.item()))

    return model, g


def validate(task_generator, forward_fn, config, model=None, xor_task=None, loss_fn=None):
    # Prepare data
    g = task_generator(config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Validate
    model.eval()
    accs = []
    losses = []
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    with torch.no_grad():
        for _ in range(config.nb_val_tasks):
            # Get and reshape data
            data_shot, label_shot, data_query, label_query = g.get_shot_query(config, device, validation=True,
                                                                              xor_task=xor_task,
                                                                              prob_xor=config.prob_xor)

            # Compute predictions
            data = torch.cat((data_shot, data_query), 0)
            embeddings_dist = model(data)
            embeddings = embeddings_dist.rsample()
            data_shot = embeddings[:data_shot.shape[0]]
            data_query = embeddings[data_shot.shape[0]:]
            probs = forward_fn(data_shot, data_query, label_shot, config=config)

            # Compute distances and loss
            loss = loss_fn(probs, label_query)  # F.cross_entropy(logits, label_query)
            acc = count_acc(probs, label_query)
            losses.append(loss)
            accs.append(acc)

    return accs, losses