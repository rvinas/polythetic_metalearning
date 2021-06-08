import torch
import torch.nn as nn



def conv_block(in_channels, out_channels):
    # bn = CustomBatchNorm()
    bn = nn.BatchNorm2d(out_channels, momentum=0.01, track_running_stats = False)
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
        self.embeddings = nn.Linear(hid_dim*3*3, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        h = x.view(x.size(0), -1)
        h = self.embeddings(h)
        # h = nn.Softplus()(h)
        return h

def count_acc(probs, label):
    pred = probs > 0.5
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

    # Train
    if loss_fn is None:
        loss_fn = nn.BCELoss()
    for epoch in range(1, config.max_epoch + 1):
        optimizer.zero_grad()
        model.train()

        # Get and reshape data
        data_shot, label_shot, data_query, label_query = g.get_shot_query(config, device, xor_task=xor_task, prob_xor=config.prob_xor)

        # Compute predictions
        data_shot = model(data_shot)
        data_query = model(data_query)
        probs = forward_fn(data_shot, data_query, label_shot, config=config)

        # Compute loss
        loss = loss_fn(probs, label_query)  # F.cross_entropy(logits, label_query)
        acc = count_acc(probs, label_query)
        loss.backward()
        optimizer.step()
        if config.verbose:
            print('epoch {}, loss={:.4f} acc={:.4f}'.format(epoch, loss.item(), acc))

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
            data_shot, label_shot, data_query, label_query = g.get_shot_query(config, device, validation=True, xor_task=xor_task)

            # Compute predictions
            data_shot = model(data_shot)
            data_query = model(data_query)
            probs = forward_fn(data_shot, data_query, label_shot, config=config)

            # Compute distances and loss
            loss = loss_fn(probs, label_query)  # F.cross_entropy(logits, label_query)
            acc = count_acc(probs, label_query)
            losses.append(loss)
            accs.append(acc)

    return accs, losses