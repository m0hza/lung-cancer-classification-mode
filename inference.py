MODEL_PATH = "path/to/ypur/model"

# Define model
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
test_dir= "test/dataset/path"
# Inference on test set
test_dataset = LungCancerDataset(test_dir, transform=transform, mode='test')
test_loader = DataLoader(test_dataset, batch_size=16)

predictions = []
with torch.no_grad():
    for images, ids in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for img_id, pred in zip(ids, predicted.cpu().numpy()):
            predictions.append((img_id, pred))

# Create CSV
submission_df = pd.DataFrame(predictions, columns=['id', 'label'])
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission file created.")
