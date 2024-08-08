import argparse

from utils import load_model, predict, load_class_names

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict flower species from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('model_path', type=str, help='Path to the trained model.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Make predictions
    probs, classes = predict(args.image_path, model, args.top_k)

    # Map classes to flower names if category_names is provided
    if args.category_names:
        class_names = load_class_names(args.category_names)
        classes = [class_names[str(cls)] for cls in classes]

    # Print the predictions
    print("Top K Predictions:")
    for i in range(len(probs)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
