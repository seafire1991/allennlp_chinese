import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """ Simple Attention

    This Attention is learned from weight
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

        # Declare the Attention Weight
        self.W = nn.Linear(dim, 1)

        # Declare the coverage feature
        self.coverage_feature = nn.Linear(1, dim)

    def forward(self, output, context, coverage):

        # declare the size
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # Expand the output to the num of timestep
        output_expand = output.expand(batch_size, input_size, hidden_size)

        # reshape to 2-dim
        output_expand = output_expand.reshape([-1, hidden_size])
        context = context.reshape([-1, hidden_size])

        # transfer the coverage to features
        coverage_feature = self.coverage_feature(coverage.reshape(-1, 1))

        # Learning the attention
        attn = self.W(output_expand + context + coverage_feature)
        attn = attn.reshape(-1, input_size)
        attn = F.softmax(attn, dim=1)
        
        # update the coverage
        coverage = coverage + attn

        context = context.reshape(batch_size, input_size, hidden_size)
        attn = attn.reshape(batch_size, -1, input_size)

        # get the value of a
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn, coverage