import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
               
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # Remove the <end> token and get the caption embeddings
        captions_embeddings = self.embed(captions[:,:-1])
        
        # Concatenate the features and the caption embeddings
        embeddings = torch.cat((features.unsqueeze(dim=1), captions_embeddings), dim=1)
        
        # Pass the embeddings through the LSTM layer, and then pass the output through the linear layer to get the scores
        lstm_out, _ = self.lstm(embeddings)        
        scores = self.linear(lstm_out)
        
        return scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Define an empty list to store the ids
        predicted_ids = []
        
        for i in range(max_len):
            # Pass the inputs through the LSTM layer, and then pass the output throught linear layer to get the scores for each word
            lstm_out, states = self.lstm(inputs, states)
            scores = self.linear(lstm_out.squeeze(dim=1))
            
            # Get the indices of the max scores and store them into the list of ids
            _, predicted = scores.max(dim=1)
            predicted_ids.append(predicted.item())
            
            # Get the inputs for the next iteration
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(dim=1)
            
            # Stop if the <end> token is predicted
            if predicted.item() == 1:
                break
        
        return predicted_ids