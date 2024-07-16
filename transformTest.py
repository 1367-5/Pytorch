import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# load pretrained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "With great power comes great"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Load pretrained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# Predict all tokens
with torch.no_grad():
  outputs = model(tokens_tensor)
  predictions = outputs[0]

# Get the predicted next subword
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)
