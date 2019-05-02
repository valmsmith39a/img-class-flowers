import argparse
import json
from model import ModelLucy

# Example: python predict.py flowers/test/1/image_06743.jpg checkpoint.pth. Correct flower type: Primerose
parser = argparse.ArgumentParser(description='Classify images of flowers - Prediction')
parser.add_argument('image_path', action="store", type=str, help="Please enter the test image path")
parser.add_argument('checkpoint_path', action="store", type=str, help="Please enter the checkpoint retrieval path")
parser.add_argument('--top_k', action="store", dest="top_k", type=int, help="Please enter top K number of most likely cases")
parser.add_argument('--category_names', action="store", dest="category_names_path", type=str, help="Please enter path to category names json")
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()

def predict():
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top_k = args.top_k or 3
    category_names_path = args.category_names_path or 'cat_to_name.json'
    use_gpu = args.gpu
    options = {
        'top_k': top_k,
        'category_names_path': category_names_path,
        'use_gpu': use_gpu
    }
    model = ModelLucy.load_checkpoint(checkpoint_path) 
    probs, classes = ModelLucy.predict(model, image_path, options)
    print('top k classes predicted: ', classes)
    print('top k probabilities predicted ', probs)

    pred_probs = probs[0].tolist()

    pred_classes = classes[0].tolist()

    # Convert indexes to flower category labels
    top_k_cat = []
    class_to_idx = model.class_to_idx

    for flower_class in pred_classes:
        category = [key for key in class_to_idx.keys() if (class_to_idx[key] == int(flower_class))][0]
        top_k_cat.append(category)

    # Convert flower category labels to names
    flower_names = []

    cat_to_name = get_cat_to_name(category_names_path)

    for category in top_k_cat:
        flower_names.append(cat_to_name[category])
            
    # Get the index of highest probability
    predicted_flower_prob = max(pred_probs)
    predicted_flower_index = pred_probs.index(predicted_flower_prob)
    predicted_flower_name = flower_names[predicted_flower_index]
    predicted_flower = {
        'predicted_flower_name': predicted_flower_name,
        'predicted_flower_prob': predicted_flower_prob
    }
    print('predicted flower name: ', predicted_flower['predicted_flower_name'])
    print('predicted flower prob: ', predicted_flower['predicted_flower_prob'])
    return predicted_flower

def get_cat_to_name(filepath):
    with open(filepath) as json_file:  
        data = json.load(json_file)
        return data

predict()

