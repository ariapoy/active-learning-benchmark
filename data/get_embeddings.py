import torch
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from torchvision import transforms
from torchvision import datasets as vision_datasets
from tqdm import tqdm

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ViT model and feature extractor
model_name = 'nateraw/vit-base-patch16-224-cifar10'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
model.eval()

# Define transformation for CIFAR-10 images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Load CIFAR-10 dataset
D_trn = vision_datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader_trn = torch.utils.data.DataLoader(D_trn, batch_size=16, shuffle=False)
D_tst = vision_datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader_tst = torch.utils.data.DataLoader(D_tst, batch_size=16, shuffle=False)

model.to(device)

# Function to extract embeddings
def extract_embeddings(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader):
            # Move images to GPU
            images = images.to(device)
            outputs = model(images)
            # Use the [CLS] token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.extend(cls_embedding)
            labels.extend(lbls.tolist())
    return embeddings, labels

# Extract embeddings
embeddings_tst, labels_tst = extract_embeddings(model, dataloader_tst)
embeddings_trn, labels_trn = extract_embeddings(model, dataloader_trn)

# Save embeddings in LibSVM format
with open('cifar10_svmstyle.txt', 'w') as f:
    for embedding, label in zip(embeddings_trn+embeddings_tst, labels_trn+labels_tst):
        features = ' '.join([f"{i+1}:{value}" for i, value in enumerate(embedding)])
        f.write(f"{label} {features}\n")

# Load pre-trained BERT model and tokenizer
model_name = 'srimeenakshiks/aspect-based-sentiment-analyzer-using-bert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()
model.to(device)

# Load IMDB dataset
dataset_trn = load_dataset('imdb', split='train')
dataset_tst = load_dataset('imdb', split='test')

# Function to extract embeddings
def extract_embeddings(model, tokenizer, texts):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Use the [CLS] token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
    return embeddings

# Extract embeddings
texts_trn = dataset_trn['text']
labels_trn = dataset_trn['label']
embeddings_trn = extract_embeddings(model, tokenizer, texts_trn)
texts_tst = dataset_tst['text']
labels_tst = dataset_tst['label']
embeddings_tst = extract_embeddings(model, tokenizer, texts_tst)

# Save embeddings in LibSVM format
with open('imdb_svmstyle.txt', 'w') as f:
    for embedding, label in zip(embeddings_trn+embeddings_tst, labels_trn+labels_tst):
        features = ' '.join([f"{i+1}:{value}" for i, value in enumerate(embedding)])
        f.write(f"{label} {features}\n")