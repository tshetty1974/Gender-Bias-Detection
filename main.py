import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Set up device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
max_seq_length = 128
batch_size = 16
num_epochs = 10
learning_rate = 2e-5

# Load and tokenize the training data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the training texts and labels
train_texts = [
    "she works hard for her age",
    "men are stronger",
    "he is a good doctor",
    "she is a good person",
    "you play well for a girl",
    "programming is not for women",
    "men are better at math",
    "women are emotional",
    "he is a skilled engineer",
    "she is a talented musician",
    "software development is dominated by men",
    "men are better leaders",
    "women are not good at sports",
    "he is a great programmer",
    "she is a caring nurse",
    "coding is a male-dominated field",
    "men are more logical than women",
    "women are too sensitive",
    "he is a successful entrepreneur",
    "she is a powerful CEO",
    "hello how are you today?",
    "today is a good day",
    "women are great leaders",
    "he is an excellent teacher",
    "she is a talented engineer",
    "men are compassionate caregivers",
    "nursing is a noble profession for both genders",
    "he is a skilled chef",
    "women excel in science and technology",
    "he is a nurturing parent",
    "she is a successful entrepreneur",
    "men are emotionally intelligent",
    "nurturing is a valuable trait in both men and women",
    "today is a good day",
    "the weather is nice",
    "I enjoy reading books",
    "exercise is important for health",
    "learning new things is exciting",
    "music brings joy to my life",
    "cooking is a creative outlet",
    "spending time with loved ones is precious",
    "nature is beautiful",
    "laughter is the best medicine",
    "you are fired because you are pregnant",
    "hey lady, you look really beautiful today!",
    "hey! Take care of the babies today",
    "Water boils at 100 degrees Celsius at sea level.",
    "The chemical formula for water is H2O."

]
train_labels = [
    "biased",
    "biased",
    "unbiased",
    "unbiased",
    "biased",
    "biased",
    "biased",
    "biased",
    "unbiased",
    "unbiased",
    "biased",
    "biased",
    "biased",
    "unbiased",
    "unbiased",
    "biased",
    "biased",
    "biased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased",
    "biased",
    "unbiased",
    "unbiased",
    "unbiased",
    "unbiased"
]

# Encode the labels as numerical values
label_map = {"biased": 0, "unbiased": 1}
train_labels = [label_map[label] for label in train_labels]

# Tokenize the texts and convert them to input features
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_seq_length, return_tensors='pt')
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Create a PyTorch DataLoader for training
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels for your task
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

# Save the fine-tuned model
model.save_pretrained('jupyter/testX')

# Inference with the fine-tuned model
model.eval()
test_texts = ["women belong to the kitchen"]  # List of input texts for inference
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_seq_length, return_tensors='pt')
test_inputs = test_encodings['input_ids'].to(device)
test_attention_mask = test_encodings['attention_mask'].to(device)

with torch.no_grad():
    logits = model(test_inputs, attention_mask=test_attention_mask).logits

predicted_labels = torch.argmax(logits, dim=1).tolist()
print('Predicted Labels:', predicted_labels)
TavishiShetty (2).pdf
Matty_s_Resume.pdf
Matty_s_Resume (1).pdf
Matty_s_Resume (2).pdf
