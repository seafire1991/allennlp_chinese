

import torch 
from .learning_attention import Attention 

class SimpleDecoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"], 
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.GRU(input_size=configure["embedding_dim"],
                                hidden_size=configure["hidden_size"],
                                num_layers=configure["num_layers"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])



    def forward(self, input, hidden):

        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding, hidden)

        out = self.fc(out.view(out.size(0),-1))

        return out, hidden



class AttentionDecoder(torch.nn.Module):
    
    def __init__(self):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=1078,
                                            embedding_dim=300)

        self.gru = torch.nn.LSTM(input_size=300+128,
                                hidden_size=128,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True)

        self.att = Attention(128)

        self.fc = torch.nn.Linear(128, 1078)

        self.p = torch.nn.Linear(30+300, 1)

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, hidden, encoder_output, z, content, coverage):

        # Embedding
        embedding = self.embedding(input)
        # print(embedding.squeeze().size())

        combine = torch.cat([embedding, z], 2)
        # print(combine.squeeze().size())
        # Call the GRU
        out, hidden = self.gru(combine, hidden)

        # call the attention
        output, attn, coverage = self.att(output=out, context=encoder_output, coverage=coverage)
        

        index = content
        attn = attn.view(attn.size(0), -1)
        attn_value = torch.zeros([attn.size(0), self.configure["num_words"]]).to(self.device)
        attn_value = attn_value.scatter_(1, index, attn)

        out = self.fc(output.view(output.size(0), -1))
        # print(torch.cat([embedding.squeeze(), combine.squeeze()], 1).size(), )
        p = self.sigmoid(self.p(torch.cat([embedding.squeeze(), combine.squeeze()], 1)))
        # print(p)
        out = (1-p)*out + p*attn_value
        # print(attn_value.size(), output.size())

        return out, hidden, output, attn, coverage
