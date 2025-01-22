import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Constants
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Sample text (Shakespeare Sonnet 2)
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# Prepare n-grams (context and target pairs)
ngrams = [
    ([test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], test_sentence[i])
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

# Vocabulary and index mapping
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}


# N-Gram Language Model
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# Training the model
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    total_loss = 0
    for context, target in ngrams:
        # Prepare the inputs
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        log_probs = model(context_idxs)

        # Compute loss
        loss = loss_function(
            log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long)
        )

        # Backward pass and update weights
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print("Training completed. Losses over epochs:", losses)

# Get embedding for a specific word (e.g., "beauty")
beauty_embedding = model.embeddings.weight[word_to_ix["beauty"]]
print("\nEmbedding for 'beauty':")
print(beauty_embedding)

# Predict next word (e.g., given context ["When", "forty"])
context = ["When", "forty"]
context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
log_probs = model(context_idxs)
predicted_word_idx = log_probs.argmax(dim=1).item()
predicted_word = ix_to_word[predicted_word_idx]
print(f"\nGiven context '{' '.join(context)}', predicted next word: {predicted_word}")
